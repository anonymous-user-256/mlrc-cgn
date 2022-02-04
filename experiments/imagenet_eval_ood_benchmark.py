"""Evaluates ResNet50 and CGN ensemble on ImageNet variants for OOD."""
import os
from os.path import join, basename, exists
import argparse
import json
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from experiments.imagenet_utils import AverageEnsembleModel
from experiments.ood_utils import (
    simple_validate,
    imagenet_adv_validate,
    stylized_imagenet_validate,
)
from cgn_framework.imagenet.models.classifier_ensemble import InvariantEnsemble


def main(args):
    """Main entrypoint function."""
    start = time.time()

    # sanity stuff
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    print("::: Loading classifier: {}".format(args.classifier))
    if args.classifier == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)

    elif args.classifier == "resnet50-from-scratch":
        # the whole point of `resnet-from-scratch` is
        # to get the same results as the original paper
        # which does not use PyTorch's checkpoint but rather one from
        # Source: https://github.com/clovaai/CutMix-PyTorch#experimental-results-and-pretrained-models
        # Referece: https://github.com/hendrycks/natural-adv-examples/issues/9

        model = torchvision.models.resnet50(pretrained=True)
        assert args.weight_path is not None, \
            "Must provide path to checkpoint for resnet50-from-scratch"
        assert basename(args.weight_path) == "resnet50_from_scratch_model_best.pth.tar", \
            "Must provide path to checkpoint for resnet50-from-scratch"

        # load model weights from checkpoint
        # args.weight_path = join(REPO_PATH, "experiments/resnet50_from_scratch_model_best.pth.tar")
        ckpt = torch.load(args.weight_path, map_location="cpu")
        ckpt_state_dict = ckpt['state_dict']
        ckpt_state_dict = {k.replace("module.", ""): v for k, v in ckpt_state_dict.items()}
        model.load_state_dict(ckpt_state_dict)

    elif args.classifier == "cgn-ensemble":
        assert args.weight_path is not None, \
            "Must provide path to checkpoint for cgn-ensemble"

        base_model = InvariantEnsemble("resnet50", pretrained=True)
        model = AverageEnsembleModel(base_model=base_model)

        # load model weights from checkpoint
        # args.weight_path = "imagenet/experiments/classifier_2022_01_19_15_36_sample_run/model_best.pth"
        ckpt = torch.load(args.weight_path, map_location="cpu")
        ckpt_state_dict = ckpt["state_dict"]
        ckpt_state_dict = {k.replace("module.", "base_model."):v for k, v in ckpt_state_dict.items()}
        model.load_state_dict(ckpt_state_dict)

    else:
        raise ValueError("Invalid classifier: {}".format(args.classifier))
    model = model.eval().to(device)
    
    # load data
    print("::: Loading dataset: {}".format(args.ood_dataset))
    data_root = join(REPO_PATH, "cgn_framework", "imagenet", "data", args.ood_dataset)
    eval_valdir = join(data_root, 'val')

    combined_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_dataset = torchvision.datasets.ImageFolder(eval_valdir, combined_transform)
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    # obtain classes (from 1000) that are present in the given dataset
    eval_classes = sorted(os.listdir(eval_valdir))
    in_mini_valdir = eval_valdir.replace(args.ood_dataset, "in-mini")
    in_mini_classes = sorted(os.listdir(in_mini_valdir))
    eval_class_indices = np.isin(in_mini_classes, eval_classes)

    # evaluate the model
    print("::: Evaluating classifier {} on {}".format(args.classifier, args.ood_dataset))
    # depending on dataset, use different validation function 
    # since different datasets define classes differently
    eval_args = dict(
        model=model,
        val_loader=eval_loader,
        gpu=device if str(device) != "cpu" else None,
        print_freq=10,
    )
    if args.ood_dataset in ["in-mini", "in-sketch"]:
        validation_fn = simple_validate
    elif args.ood_dataset == "in-a":
        validation_fn = imagenet_adv_validate
        eval_args.update(dict(eval_indices=eval_class_indices))
    elif args.ood_dataset == "in-stylized":
        validation_fn = stylized_imagenet_validate
    else:
        raise ValueError("Invalid dataset: {}".format(args.ood_dataset))

    acc1 = validation_fn(**eval_args)
    acc1 = np.round(acc1, decimals=3)
    print("::: Summary: {},{},{}".format(args.classifier, args.ood_dataset, acc1))

    end = time.time()
    print("::: Elapsed time: {:.2f}s".format(end - start))

    return acc1


def eval_ood(args, show=False):

    run_name = f"{args.classifier}_{args.ood_dataset}"
    result_dir = join(REPO_PATH, "cgn_framework/imagenet/experiments/ood_eval", run_name)
    os.makedirs(result_dir, exist_ok=True)
    result_path = join(result_dir, f"results_seed_{args.seed}.json")

    if exists(result_path):
        if not args.ignore_cache:
            print("::: Result file {} already exists & --ignore_cache={}".format(
                result_path, args.ignore_cache
            ))
            with open(result_path, "r") as f:
                result = json.load(f)
            
            if show:
                print("::: Result: {}".format(json.dumps(result, indent=4)))
            
            return result
        else:
            print("::: Result file {} already exists, but --ignore_cache={}".format(
                result_path, args.ignore_cache
            ))
            print("::: Re-running evaluation")

    if args.weight_path is not None:
        args.weight_path = join(REPO_PATH, args.weight_path)
        assert exists(args.weight_path), \
            "Weight path {} does not exist".format(args.weight_path)

    acc1 = main(args)

    print("::: Saving results to {}.".format(result_path))
    result = dict(
        **vars(args),
        acc1=acc1,
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, required=True,
                        choices=["cgn-ensemble", "resnet50", "resnet50-from-scratch"],
                        help='Provide classifer to be evaluated.')
    parser.add_argument('--ood_dataset', type=str, required=True,
                        choices=["in-mini", "in-a", "in-stylized", "in-sketch"],
                        help='Provide dataset name.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='path to the classifier checkpoint relative to REPO_PATH')
    parser.add_argument('--ignore_cache', action='store_true',
                        help='ignore cache and force re-evaluation')
    args = parser.parse_args()

    print(args)

    result_dir = join(REPO_PATH, "experiments", "results", "cache")
    result_path = join(result_dir, "ood_{}_{}.json".format(args.classifier, args.ood_dataset))
    if exists(result_path):
        if not args.ignore_cache:
            print("::: Result file {} already exists & --ignore_cache={}".format(
                result_path, args.ignore_cache
            ))
            with open(result_path, "r") as f:
                result = json.load(f)
            print("::: Result: {}".format(json.dumps(result, indent=4)))
            exit()
        else:
            print("::: Result file {} already exists, but --ignore_cache={}".format(
                result_path, args.ignore_cache
            ))
            print("::: Re-running evaluation")

    if args.weight_path is not None:
        args.weight_path = join(REPO_PATH, args.weight_path)
        assert exists(args.weight_path), \
            "Weight path {} does not exist".format(args.weight_path)

    acc1 = main(args)

    print("::: Saving results to {}.".format(result_path))
    result = dict(
        **vars(args),
        acc1=acc1,
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

