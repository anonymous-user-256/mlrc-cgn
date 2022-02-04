"""Evaluates model performance on the imagenet-sketch benchmark."""
import re
import os
from os.path import join, basename, dirname
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision import transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from experiments.image_utils import denormalize, show_single_image
from experiments.imagenet_utils import IMModel, AverageEnsembleModel
from experiments.ood_utils import validate as ood_validate
from cgn_framework.imagenet.models.classifier_ensemble import InvariantEnsemble


if __name__ == "__main__":
    seed_everything(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    model = InvariantEnsemble("resnet50", pretrained=True)
    ckpt_path = "imagenet/experiments/classifier_2022_01_19_15_36_sample_run/model_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = ckpt["state_dict"]
    ckpt_state_dict = {k.replace("module.", ""):v for k, v in ckpt_state_dict.items()}
    model.load_state_dict(ckpt_state_dict)
    model = model.eval().to(device)

    shape_model = IMModel(base_model=model, mode="shape")
    avg_model = AverageEnsembleModel(base_model=model)
    pytorch_model = torchvision.models.resnet50(pretrained=True).to(device)
    # net = torch.hub.load('pytorch/vision:v0.10.0', "resnet50", pretrained=True)
    ckpt = torch.load(join(REPO_PATH, "experiments/resnet50_from_scratch_model_best.pth.tar"), map_location="cpu")
    ckpt_state_dict = ckpt['state_dict']
    ckpt_state_dict = {k.replace("module.", ""): v for k, v in ckpt_state_dict.items()}
    pytorch_model.load_state_dict(ckpt_state_dict)

    # pytorch_model = torchvision.models.densenet121(pretrained=True).to(device)
    only_backbone_model = torch.nn.Sequential(model.backbone, torch.nn.Flatten(), torch.nn.Linear(2048, 1000)).to(device)

    # check models
    x = torch.randn((1, 3, 224, 224)).to(device)

    y = shape_model(x)
    assert y.shape == torch.Size([1, 1000])

    y = avg_model(x)
    assert y.shape == torch.Size([1, 1000])

    y = pytorch_model(x)
    assert y.shape == torch.Size([1, 1000])

    y = only_backbone_model(x)
    assert y.shape == torch.Size([1, 1000])

    # load dataset
    # valdir = join(join(REPO_PATH, "cgn_framework", "imagenet", "data/sketch"), 'val')
    valdir = join(join(REPO_PATH, "cgn_framework", "imagenet", "data/in-a"), 'val')
    eval_classes = sorted(os.listdir(valdir))

    in_mini_dir = join(join(REPO_PATH, "cgn_framework", "imagenet", "data/in-mini"), 'val')
    in_mini_classes = sorted(os.listdir(in_mini_dir))

    eval_class_indices = np.isin(in_mini_classes, eval_classes)
    # import ipdb; ipdb.set_trace()


    # target transform for imagenet-a dataset
    metadata_path = join(REPO_PATH, "cgn_framework", "imagenet", "data", "in-mini", "metadata.txt")
    with open(metadata_path, "r") as f:
        metadata = f.readlines()
    in1k_mapping = {line.split(" ")[0]: int(line.split(" ")[1]) - 1 for line in metadata}
    inad_mapping = dict(enumerate(sorted(os.listdir(valdir))))

    import json
    with open(join(REPO_PATH, "cgn_framework/imagenet/data/in-a/classes_to_select.json")) as f:
        thousand_k_to_200 = json.load(f)

    # import ipdb; ipdb.set_trace()
    target_transform = None
    # if "in-a" in valdir:
    #     target_transform = transforms.Lambda(lambda x: in1k_mapping[inad_mapping[x]])
    # print(sorted([in1k_mapping[inad_mapping[x]] for x in range(len(os.listdir(valdir)))]))
    print([i for i, flag in enumerate(eval_class_indices) if flag])
    print([int(k) for k in thousand_k_to_200 if thousand_k_to_200[k] != -1])

    # valdir = join(join(REPO_PATH, "cgn_framework", "imagenet", "data/stylized-imagenet"), 'val')
    combined_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, combined_transform, target_transform=target_transform,),
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    print("::::::: Evaluating PyTorch ResNet50 model trained on ImageNet :::::::")
    # acc1 = ood_validate(val_loader, pytorch_model, gpu=device)
    # acc1 = ood_validate(val_loader, shape_model, gpu=device)
    # acc1 = ood_validate(val_loader, avg_model, gpu=device)
    # acc1 = ood_validate(val_loader, only_backbone_model, gpu=device)

    # from ood_utils import stylized_imagenet_validate
    # print("Pytorch ResNet50")
    # stylized_imagenet_validate(val_loader, pytorch_model, gpu=device)
    # print("Shape model")
    # stylized_imagenet_validate(val_loader, shape_model, gpu=device)
    # print("Average model")
    # stylized_imagenet_validate(val_loader, avg_model, gpu=device)

    indices_in_1k = [int(k) for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]
    from ood_utils import imagenet_adv_validate
    # acc1 = imagenet_adv_validate(val_loader, pytorch_model, gpu=device, eval_indices=eval_class_indices)
    acc1 = imagenet_adv_validate(val_loader, avg_model, gpu=device, eval_indices=eval_class_indices)

    