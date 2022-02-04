"""
Performs experiments on IN-mini dataset.
This primarily replicates the numbers of Table 3 and 4 in the paper
"""
import json
import os
from os.path import join, exists, isdir
from subprocess import call
from glob import glob
import pandas as pd
import torch

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything, dotdict
set_env()

from imagenet_eval_ood_benchmark import eval_ood


def generate_counterfactual_dataset(
        prefix,
        modes=["train",
        "val"],
        trunc=0.5,
        n_train=34745,
        n_val=3923,
        seed=0,
    ):
    """Generates CF dataset for ImageNet (size of IN-mini)"""
    seed_everything(seed)

    script_path = join(REPO_PATH, "cgn_framework/imagenet/generate_data.py")

    # generate train and val dataset
    for mode in modes:        
        run_name = f"{prefix}_{mode}_trunc_{trunc}"
        n_samples = eval(f"n_{mode}")

        data_root = join(REPO_PATH, "cgn_framework/imagenet/data", run_name)
        ims = glob(join(data_root, "ims/*.jpg"))

        if isdir(data_root) and len(ims) == n_samples:
            print("")
            print(f"{mode.capitalize()} dataset exists with {n_samples} images, skipping...")
            print(f"Path to dataset: {data_root}")
            print("")
        else:
            print("Generating {} dataset...".format(mode))
            print("WARNING: This will take about 3 hours for train set and 20 mins for validation set.")
            arguments = "--mode random --weights_path imagenet/weights/cgn.pth"\
                f" --n_data {n_samples} --run_name {prefix}-{mode} --truncation {trunc} --batch_sz 1"\
                f" --ignore_time_in_filename"
            cmd = f"python {script_path} {arguments}"
            call(cmd, shell=True)


def train_classifier(args: dict = dict(lr=0.001), prefix="in-mini", seed=0, disp_epoch=45, show=False, ignore_cache=False):
    """Trains classifier on IN-mini dataset"""

    args = dotdict(args)
    seed_everything(seed)

    run_name = f"{prefix}-classifier"
    expt_dir = join(REPO_PATH, "cgn_framework/imagenet/experiments", f"classifier__{run_name}")
    epoch_metrics_path = join(expt_dir, f"epochwise_metrics/epoch_{disp_epoch}.pt")
    if not exists(epoch_metrics_path) or ignore_cache:
        
        print("::::: Training classifier :::::")
        script_path = join(REPO_PATH, "cgn_framework/imagenet/train_classifier.py")

        # all arguments used are defaults given in their repo/paper
        arguments = f"-a resnet50 -b 32 --lr {args.lr} -j 6 --pretrained"\
            f" --data imagenet/data/in-mini --cf_data imagenet/data/{prefix}"\
            f" --name {run_name} --seed {seed} --ignore_time_in_filename"
        cmd = f"python {script_path} {arguments}"
        call(cmd, shell=True)
    
    else:
        print("::::: Classifier already trained, skipping :::::")

    print(f"Loading results for epoch {disp_epoch} from {epoch_metrics_path}")
    metrics = torch.load(epoch_metrics_path)
    if show:
        print(json.dump(metrics, indent=4))
    
    return metrics


def run_eval_on_ood_benchmarks(seed=0, ignore_cache=False, show=False):
    classifiers=["cgn-ensemble", "resnet50"]
    datasets=["in-mini", "in-a", "in-stylized", "in-sketch"]

    df = pd.DataFrame(columns=datasets, index=classifiers)

    for classifier in classifiers:
        for dataset in datasets:
            print(f"::::: Running {classifier} on {dataset}...")

            weight_path = None
            if dataset == "in-a" and classifier == "resnet50":
                # modify temporarily
                classifier = "resnet50-from-scratch"
                # this should be downloaded from a script in setup
                weight_path = "cgn_framework/imagenet/weights/resnet50_from_scratch_model_best.pth.tar"
            
            if classifier == "cgn-ensemble":
                weight_path = "cgn_framework/imagenet/weights/classifier_on_in-mini_model_best.pth"

            args = dict(
                seed=seed,
                classifier=classifier,
                ood_dataset=dataset,
                num_workers=2,
                batch_size=128,
                ignore_cache=ignore_cache,
            )
            if weight_path is not None:
                args["weight_path"] = weight_path
            args = dotdict(args)
            result = eval_ood(args, show=show)

            if dataset == "in-a" and classifier == "resnet50-from-scratch":
                classifier = "resnet50"

            df.at[classifier, dataset] = result["acc1"]

    return df


def run_experiments(seed=0, generate_cf_data=False, disp_epoch=45, ignore_cache=False):
    """Runs experiments on IN-mini dataset

    1. Generates CF dataset
    2. Runs classifier experiments
    """
    seed_everything(seed)

    # step 1: generate dataset
    if generate_cf_data:
        print("WARNING: You have passed generate_cf_data=True.")
        print("WARNING: This will take about 3 hours for train set and 20 mins for validation set.")
        print("\n::::: Generating CF dataset :::::\n")
        generate_counterfactual_dataset(prefix="in-mini", seed=seed)
    else:
        print("Since generate_cf_data=False, skipping CF dataset generation.")
        print("Loading results for classification and OOD experiments from cache.")

    # step 2: train classifier
    print("\n::::: Training classifier :::::\n")
    metrics = train_classifier(prefix="in-mini", seed=seed, disp_epoch=disp_epoch, ignore_cache=ignore_cache)

    # step 3: evaluate on OOD benchmarks
    print("\n::::: Evaluating OOD :::::\n")
    df_ood = run_eval_on_ood_benchmarks(seed=seed, show=False, ignore_cache=ignore_cache)

    return metrics, df_ood



if __name__ == "__main__":
    metrics_clf, df_ood = run_experiments(seed=0, generate_cf_data=False, disp_epoch=34, ignore_cache=False)

    # construct Table 3 of the paper
    heads = ["shape", "texture", "bg"]
    table_3 = pd.DataFrame(
        None,
        columns=["Shape bias", "Top 1", "Top 5"],
        index=[f"IN-mini + CGN/{h}" for h in heads],
    )
    for i, h in enumerate(heads):
        table_3.at[f"IN-mini + CGN/{h}", "Shape bias"] = metrics_clf[f"shape_biases/{i}_m_{h}_bias"]
        table_3.at[f"IN-mini + CGN/{h}", "Top 1"] = metrics_clf[f"acc1/1_real"]
        table_3.at[f"IN-mini + CGN/{h}", "Top 5"] = metrics_clf[f"acc5/1_real"]

    table_3["Shape bias"] *= 100.0
    table_3 = table_3.astype(float).round(1)
    print("\n::::: Table 3 :::::")
    print(table_3)
    print()

    # construct Table 4 of the paper
    table_4 = pd.DataFrame(
        None,
        columns=["IN-9", "Mixed-same", "Mixed-rand", "BG-gap"],
        index=["IN-mini + CGN"],
    )

    col_to_key = {
        "IN-9": "in_9_acc1_original/shape_texture",
        "Mixed-same": "in_9_acc1_mixed_same/shape_texture",
        "Mixed-rand": "in_9_acc1_mixed_rand/shape_texture",
        "BG-gap": "in_9_gaps/bg_gap",
    }

    for c in table_4.columns:
        assert col_to_key[c] in metrics_clf
        key = col_to_key[c]
        table_4.at["IN-mini + CGN", c] = metrics_clf[key]

    table_4 = table_4.astype(float).round(1)
    print("\n::::: Table 4 :::::")
    print(table_4)
    print()

    print("\n::::: Table 5 :::::")
    print(df_ood)
    print()