"""
Script that runs the following on MNIST variants multiple times with different seeds:
1. Generates counterfactual data using given checkpoint
2. Trains a classifier on the counterfactual data
3. Computes Grad-CAM on the test set and IoU metric
"""
import os
from subprocess import call
import argparse

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from cgn_framework.mnists.dataloader import TENSOR_DATASETS



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=TENSOR_DATASETS,
                        help='Provide dataset name.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 10, 20, 30, 40],
                        help='random seeds')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='path to CGN pretrained checkpoint')
    args = parser.parse_args()

    print("::::: Configs :::::")
    print(args)

    for seed in args.seeds:
        print(f"\n::::::: Running with seed {seed} :::::::")
        seed_everything(seed)

        # generate counterfactual data
        cmd = f"python {REPO_PATH}/cgn_framework/mnists/generate_data.py --dataset {args.dataset} --weight_path {args.weight_path}"
        print(f"\n::: Generating counterfactual data :::")
        print(cmd)
        call(cmd, shell=True)

        # train classifier on counterfactual data
        cmd = f"python {REPO_PATH}/cgn_framework/mnists/train_classifier.py --dataset {args.dataset}_counterfactual --seed {seed}"
        print(f"\n::: Training classifier on counterfactual data :::")
        print(cmd)
        call(cmd, shell=True)

        # compute Grad-CAM on test set
        weight_path = f"mnists/experiments/classifier_{args.dataset}_counterfactual_seed_{seed}/weights/ckp_epoch_10.pth"
        cmd = f"python {REPO_PATH}/experiments/quant_metric_for_gradcam.py --dataset {args.dataset} --weight_path {weight_path} --seed {seed} --disable_tqdm"
        print(f"\n::: Computing Grad-CAM on test set :::")
        print(cmd)
        call(cmd, shell=True)
