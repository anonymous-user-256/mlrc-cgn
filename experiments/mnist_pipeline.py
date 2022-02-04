"""Defines a pipeline for an end-to-end experiment on MNISTs."""
import os
import sys
import argparse
import json
from glob import glob
from subprocess import call
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torchvision.datasets as tv_datasets

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything, dotdict
set_env()

from cgn_framework.mnists.dataloader import TENSOR_DATASETS
from cgn_framework.mnists.config import get_cfg_defaults
from cgn_framework.utils import load_cfg


class MNISTPipeline:
    """
    Pipeline to run *a single experiment* on an MNIST variant.

    Step 1: (Optional) Train a GAN/CGN model on the given dataset.
    Step 2: (Optional) Generate data using trained/given GAN/CGN/None model.
            In case of None, tensor for original dataset is generated.
    Step 3: Train a classifier on the generated data.

    Each Step saves the results to a directory and is not run if cached results exist.
    If generate=False, the pipeline will run for original dataset.
    """
    def __init__(self,
            args,
            setting="Original",
            train_generative=False,
            generate=True,
            ignore_cache=False,
        ) -> None:
        """
        Initialize the pipeline.

        Args:
            args: Arguments for the experiment.
            train_generative: Whether to train a GAN/CGN model.
            generate: Whether to generate data.
            ignore_cache: Whether to ignore cached results.
        """
        self.train_generative = train_generative
        self.generate = generate
        self.setting = setting
        self.ignore_cache = ignore_cache
        self.args = self._check_args(args)
        print("::::: Experimental setup :::::")
        print("Train generative model:", self.train_generative)
        print("Generate data:", self.generate)
        print("Args:", self.args)

    def _check_args(self, args):
        """Check the arguments."""
        assert isinstance(args.seed, int)
        if args.dataset not in TENSOR_DATASETS:
            raise ValueError("The dataset is not supported.")

        if self.train_generative:
            assert hasattr(args, "cfg") and args.cfg is not None, \
                "You need to pass config file for training GAN/CGN."\
                    "If not, set train_generative to False."
            
            # check if given dataset matches the given config
            assert args.dataset in args.cfg, \
                "The dataset in args.dataset does not match the dataset in args.cfg."

            # NOTE: this is config used only for training generative model
            self.gencfg = load_cfg(args.cfg) if args.cfg else get_cfg_defaults()
    
        if self.generate and not self.train_generative and self.setting != "Original":
            if not os.path.exists(args.weight_path):
                raise FileNotFoundError(
                    "The weight path for given CGN/GAN model does not exist."
                )
        
        if not self.generate and not self.train_generative:
            args.combined = False

        return args

    def train_generative_model(self):
        """Train a GAN/CGN model on the given dataset and returns path to best ckpt."""

        if self.train_generative:
            generative_model = "gan" if "gan" in self.args.cfg else "cgn"
            cmd = f"python {REPO_PATH}/cgn_framework/mnists/train_{generative_model}.py "\
                f"--cfg {self.args.cfg} --ignore_time_in_filename"
            print(f"Running command: {cmd}")
            call(cmd, shell=True)
        
            ckpt_dir =  os.path.join(
                REPO_PATH,
                "cgn_framework/mnists/experiments",
                f"cgn_{self.gencfg.TRAIN.DATASET}__{self.gencfg.MODEL_NAME}",
                "weights"
            )
            weight_path = glob(os.path.join(ckpt_dir, "*.pth"))[-1]
        else:
            weight_path = self.args.weight_path

        return weight_path
    
    def generate_data(self, dataset, weight_path, dataset_suffix=""):
        """Generate data using trained/given GAN/CGN/None model."""
        if self.generate:

            if self.setting == "Original":
                train_file_path = os.path.join(
                    REPO_PATH,
                    f"cgn_framework/mnists/data/{dataset}_train.pth",
                )
                test_file_path = train_file_path.replace("_train", "_test")
                results_exist = (not (os.path.exists(train_file_path) and os.path.exists(test_file_path)))
                if results_exist or self.ignore_cache:
                    cmd = f"python {REPO_PATH}/cgn_framework/mnists/generate_data.py --dataset {dataset}"
                    print(cmd)
                    call(cmd, shell=True)
                else:
                    print(f"Train Dataset already generated at {train_file_path}.")
                    print(f"Test Dataset already generated at {test_file_path}.")
            else:
                tensor_file_path = os.path.join(
                    REPO_PATH,
                    f"cgn_framework/mnists/data/{dataset}{dataset_suffix}.pth",
                )
                if (not os.path.exists(tensor_file_path)) or self.ignore_cache:
                    cmd = f"python {REPO_PATH}/cgn_framework/mnists/generate_data.py"\
                        f" --dataset {dataset}  --weight_path {weight_path}"
                    print(cmd)
                    call(cmd, shell=True)
                else:
                    print(f"Dataset already generated at {tensor_file_path}.")
    
    def train_classifier(self, seed, dataset, dataset_suffix="", combined=False):
        """Train a classifier on the generated data."""
        dataset += dataset_suffix

        # extract classifier results
        expt_suffix = (dataset) if not combined else (dataset + "_combined")
        expt_suffix += "_seed_" + str(seed) if seed is not None else ""

        train_results_path = f'mnists/experiments/classifier_{expt_suffix}/train_accs.pth'
        test_results_path = f'mnists/experiments/classifier_{expt_suffix}/test_accs.pth'

        results_exist = not (os.path.exists(test_results_path) and os.path.exists(train_results_path))
        if results_exist or self.ignore_cache:
            cmd = f"python {REPO_PATH}/cgn_framework/mnists/train_classifier.py"\
                f" --dataset {dataset} --seed {seed}" + \
                (f" --combined" if combined else "")
            print(cmd)
            call(cmd, shell=True)
        else:
            print(f"Results for classifier already exist: {test_results_path}")

        train_results = torch.load(train_results_path)
        test_results = torch.load(test_results_path)
        return {"train": train_results, "test": test_results}

    def run(self):
        """Main experiment runner."""
        seed_everything(self.args.seed)

        # train generative model
        weight_path = self.train_generative_model()

        # generate data
        dataset_suffix = ""
        if self.setting != "Original":
            dataset_suffix = "_counterfactual" if f"cgn_{self.args.dataset}" in weight_path else "_gan"
        self.generate_data(self.args.dataset, weight_path, dataset_suffix=dataset_suffix)

        # train classifier
        results = self.train_classifier(
            self.args.seed, self.args.dataset, dataset_suffix, self.args.combined,
        )
        return results


def run_experiments(
        seed=0,
        datasets=["colored_MNIST", "double_colored_MNIST", "wildlife_MNIST"],
        show=False,
        ignore_cache=False,
    ):
    """Run experiments on MNISTs"""

    columns = []
    for d in datasets:
        columns.extend([d + '-train', d + "-test"])
    rows = [
        "Original",
        "Original + GAN",
        "Original + CGN",
        "Original + GAN (combined)",
        "Original + CGN (combined)",
    ]
    df = pd.DataFrame(None, columns=columns)

    for dataset in datasets:

        # training on original dataset
        pipeline = MNISTPipeline(
            args=dotdict(dict(seed=seed, dataset=dataset)),
            train_generative=False,
            generate=True,
            setting=rows[0],
            ignore_cache=ignore_cache,
        )
        result = pipeline.run()
        # here: 10 denotes the last epoch
        df.at[rows[0], dataset + "-train"] = result["train"][10]
        df.at[rows[0], dataset + "-test"] = result["test"][10]

        # generate GAN dataset -> training on GAN dataset (not combined)
        pipeline = MNISTPipeline(
            args=dotdict(
                dict(
                    seed=seed,
                    dataset=dataset,
                    weight_path=os.path.join(
                        REPO_PATH,
                        f"cgn_framework/mnists/experiments/gan_{dataset}/weights/ckp.pth",
                    ),
                )
            ),
            train_generative=False,
            generate=True,
            setting=rows[1],
            ignore_cache=ignore_cache,
        )
        result = pipeline.run()
        df.at[rows[1], dataset + "-train"] = result["train"][10]
        df.at[rows[1], dataset + "-test"] = result["test"][10]

        # generate CGN dataset -> training on CGN dataset (not combined)
        pipeline = MNISTPipeline(
            args=dotdict(
                dict(
                    seed=seed,
                    dataset=dataset,
                    weight_path=os.path.join(
                        REPO_PATH,
                        f"cgn_framework/mnists/experiments/cgn_{dataset}/weights/ckp.pth",
                    ),
                )
            ),
            train_generative=False,
            generate=True,
            setting=rows[2],
            ignore_cache=ignore_cache,
        )
        result = pipeline.run()
        df.at[rows[2], dataset + "-train"] = result["train"][10]
        df.at[rows[2], dataset + "-test"] = result["test"][10]

        # generate GAN dataset -> training on GAN dataset (combined)
        pipeline = MNISTPipeline(
            args=dotdict(
                dict(
                    seed=seed,
                    dataset=dataset,
                    weight_path=os.path.join(
                        REPO_PATH,
                        f"cgn_framework/mnists/experiments/gan_{dataset}/weights/ckp.pth",
                    ),
                    combined=True,
                )
            ),
            train_generative=False,
            generate=True,
            setting=rows[2],
            ignore_cache=ignore_cache,
        )
        result = pipeline.run()
        df.at[rows[3], dataset + "-train"] = result["train"][10]
        df.at[rows[3], dataset + "-test"] = result["test"][10]

        # generate CGN dataset -> training on CGN dataset (combined)
        pipeline = MNISTPipeline(
            args=dotdict(
                dict(
                    seed=seed,
                    dataset=dataset,
                    weight_path=os.path.join(
                        REPO_PATH,
                        f"cgn_framework/mnists/experiments/cgn_{dataset}/weights/ckp.pth",
                    ),
                    combined=True,
                )
            ),
            train_generative=False,
            generate=True,
            setting=rows[2],
            ignore_cache=ignore_cache,
        )
        result = pipeline.run()
        df.at[rows[4], dataset + "-train"] = result["train"][10]
        df.at[rows[4], dataset + "-test"] = result["test"][10]

    print("")
    if show:
        print("::::::::::::::::::: Final result :::::::::::::::::::")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
        print("::::::::::::::::::::::::::::::::::::::::::::::::::::")
    else:
        return df


if __name__ == "__main__":
    df = run_experiments(seed=0, ignore_cache=False)
    df.index = ["Original", "GAN", "CGN", "Original + GAN", "Original + CGN"]
    print("::: Displaying Table 2 :::\n")
    print(df.astype(float).round(1))
