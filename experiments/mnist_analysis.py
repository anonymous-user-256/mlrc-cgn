"""Performs additional analysis on MNIST variants."""
import re
import os
from os.path import join
import json
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from tqdm import tqdm
from sklearn.manifold import TSNE

import torch
from torchvision.utils import save_image
from torchvision.io import read_image
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from cgn_framework.mnists.models.classifier import CNN
from cgn_framework.mnists.train_cgn import save
from cgn_framework.mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS
from experiments.gradcam_utils import compute_iou_between_gt_and_gradcam_for_mnist


def get_model_features(model, dl, device, num_batches_to_use=None):
    iterator = tqdm(
        dl,
        desc="Extracting features",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )
    fvecs = []
    labels = []
    ctr = 0
    for (data, label) in iterator:
        data = data.to(device)
        label = label.to(device)

        fvec = model(data)

        fvecs.append(fvec.cpu())
        labels.append(label.cpu())
        
        if ctr == num_batches_to_use:
            break
        
        ctr += 1

    fvecs = torch.cat(fvecs, dim=0)
    labels = torch.cat(labels, dim=0)
    return fvecs, labels


def reduce_dimensionality(X, dim=2):
    assert len(X.shape) == 2
    N, D = X.shape
    assert D >= dim
    
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    tsne = TSNE(n_components=dim)
    Z = tsne.fit_transform(X)
    return Z


def create_df(Z, y):
    df = pd.DataFrame(None)
    df["Z1"] = Z[:, 0]
    df["Z2"] = Z[:, 1]
    df["y"] = y
    return df


def plot_features(
        df_original,
        df_counterfactual,
        dataset="colored_MNIST",
        model_desc="CNN classifier",
        save=True,
        show=False,
        set_title=True,
    ):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    ax[0].grid()
    ax[0].set_title("Original", fontsize=20)

    colors = ["red", "cyan", "yellow", "skyblue", "orange", "purple", "cyan", "pink", "limegreen", "salmon"]
    sns.scatterplot(data=df_original, x="Z1", y="Z2", hue="y", ax=ax[0], palette="deep")
    ax[0].legend(fontsize=16)

    ax[1].grid()
    ax[1].set_title("Counterfactual", fontsize=20)
    sns.scatterplot(data=df_counterfactual, x="Z1", y="Z2", hue="y", ax=ax[1], palette="deep")
    ax[1].legend(fontsize=16)

    if set_title:
        plt.suptitle(f"Features for  {model_desc} ({dataset.replace('_', ' ')})", fontsize=25)

    if save:
        save_path = join(
            REPO_PATH,
            f"experiments/results/plots/feature_analysis_{model_desc.replace(' ', '_')}_{dataset}.pdf",
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, bbox_inches="tight", format="pdf")

    if show:
        plt.show()


def plot_iou_for_og_and_cf(og_path_per_dataset, cf_path_per_dataset, save=False, show=False):

    # construct df
    # df = pd.DataFrame(None)
    df = []
    for dataset in og_path_per_dataset.keys():
        og_path = og_path_per_dataset[dataset]
        cf_path = cf_path_per_dataset[dataset]
        iou_og = torch.load(og_path)
        iou_cf = torch.load(cf_path)

        df.append([dataset, "Original", iou_og["overall_mean"]])
        df.append([dataset, "Counterfactual", iou_cf["overall_mean"]])
        # df["iou"] = [iou_og["overall_mean"], iou_cf["overall_mean"]]
        # df["dataset"] = ["colored_MNIST", "colored_MNIST"]
        # df["model"] = ["Original", "Counterfactual"]
    df = pd.DataFrame(df, columns=["dataset", "model", "iou"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.grid()
    ax.set_title("IoU of GradCAM heatmaps with ground truth", fontsize=22)

    # sns.barplot(data=df, x="dataset", y="iou", hue="model", palette="Paired")
    sns.barplot(data=df, x="dataset", y="iou", hue="model", palette=["skyblue", "pink"])

    ax.legend(bbox_to_anchor=(0.3, 0.9), fontsize=15, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(ax.get_xlabel().capitalize(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel().upper(), fontsize=18)
    ax.set_ylim(0, 0.5)

    if save:
        plt.savefig(
            join(REPO_PATH, "experiments", "results", "quant_gradcam_iou.pdf"),
            bbox_inches="tight",
        )
    
    if show:
        plt.show()


def show_gradcam_qualitative(dataset, paths_og, paths_cf):
    from PIL import Image

    fig, axes = plt.subplots(len(paths_og), 2, figsize=(10, 2 * len(paths_og)), constrained_layout=True)
    for i, (og_path, cf_path) in enumerate(zip(paths_og, paths_cf)):
        ax = axes[i][0]
        ax.set_title(f"Original", fontsize=16)
        ax.grid()
        ax.imshow(np.squeeze(np.array(Image.open(og_path))))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[i][1]
        ax.set_title(f"Counterfactual", fontsize=16)
        ax.grid()
        ax.imshow(np.squeeze(np.array(Image.open(cf_path))))
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f"GradCAM heatmaps for {dataset.replace('_', ' ')}", fontsize=20)
    plt.show()


class MNISTAnalysis:
    """
    Performs additional analyses on MNISTs.

    1. Visualizes features using t-SNE
    2. Visualize Grad-CAM on test set and compute IoU metric
    """
    def __init__(self, dataset, weight_path, seed=0, ignore_cache=False) -> None:
        self._check_args(dataset, weight_path, seed)
        self.dataset = dataset
        self.weight_path = weight_path
        self.seed = seed
        self.ignore_cache = ignore_cache

        seed_everything(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _check_args(self, dataset, weight_path, seed):
        """Checks arguments"""
        assert dataset in TENSOR_DATASETS, f"{dataset} is not a valid dataset"
        assert os.path.exists(weight_path), f"{weight_path} does not exist"
        assert isinstance(seed, int)
    
    def visualize_feature(self, num_batches_to_use=20, show=False, save=True):

        save_dir = os.path.dirname(self.weight_path).replace("weights", "features")
        os.makedirs(save_dir, exist_ok=True)
        save_path_og = join(save_dir, f"{self.dataset}_original.pth")
        save_path_cf = join(save_dir, f"{self.dataset}_counterfactual.pth")

        if not os.path.exists(save_path_og) or not os.path.exists(save_path_cf) or self.ignore_cache:

            # load dataloaders
            print("Loading datasets...")
            dl_og_train, dl_og_test = get_tensor_dataloaders(dataset=f"{self.dataset}")
            dl_cf_train, dl_cf_test = get_tensor_dataloaders(dataset=f"{self.dataset}_counterfactual")

            # load model
            print("Loading model...")
            model = CNN()
            model.load_state_dict(torch.load(self.weight_path, map_location="cpu"))
            model.cls = torch.nn.Identity()
            model = model.eval().to(self.device)

            # get features
            features_og, y_og = get_model_features(
                model, dl_og_test, self.device, num_batches_to_use=num_batches_to_use,
            )
            print("Original features extracted of shape {}".format(features_og.shape))
            features_cf, y_cf = get_model_features(
                model, dl_cf_test, self.device, num_batches_to_use=num_batches_to_use,
            )
            print("Counterfactual features extracted of shape {}".format(features_cf.shape))

            # reduce dimensionality
            print("Reducing dimensionality...")
            Z_og = reduce_dimensionality(features_og)
            Z_cf = reduce_dimensionality(features_cf)

            print("Saving features at ...")
            print("Original: {}".format(save_path_og))
            print("Counterfactual: {}".format(save_path_cf))
            torch.save({"feat": features_og, "y": y_og, "tsne": Z_og}, save_path_og)
            torch.save({"feat": features_cf, "y": y_cf, "tsne": Z_cf}, save_path_cf)
        else:
            print(f"Loading saved features from {save_path_og} and {save_path_cf}")
            Z_og = torch.load(save_path_og, map_location="cpu")["tsne"]
            Z_cf = torch.load(save_path_cf, map_location="cpu")["tsne"]
            y_og = torch.load(save_path_og, map_location="cpu")["y"]
            y_cf = torch.load(save_path_cf, map_location="cpu")["y"]

        # create dataframe
        df_og = create_df(Z_og, y_og)
        df_cf = create_df(Z_cf, y_cf)

        # plot
        model_type = 'original' if not 'counterfactual' in self.weight_path else 'counterfactual'
        plot_features(
            df_og,
            df_cf,
            dataset=self.dataset,
            model_desc=f"CNN classifier trained on {model_type}",
            save=save,
            show=show,
            set_title=False,
        )
    
    def perform_gradcam_quantitative(self, debug=False):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(self.weight_path)), "gradcam")
        os.makedirs(save_dir, exist_ok=True)
        save_path = join(save_dir, f"{self.dataset}_gradcam_iou_seed_{self.seed}.pth")

        if not os.path.exists(save_path) or self.ignore_cache:
            class_wise_iou, sample_indices, samples = compute_iou_between_gt_and_gradcam_for_mnist(
                dataset=self.dataset,
                weight_path=self.weight_path,
                seed=self.seed,
                debug=debug,
                return_samples=True,
            )
            print("Saving gradcam results at {}".format(save_path))
            torch.save(class_wise_iou, save_path)

            # save samples
            print(f"Saving samples: {len(samples)}")
            for i, sample in zip(sample_indices, samples):
                save_path = join(save_dir, f"{self.dataset}_gradcam_seed_{self.seed}_sample_{i}.png")
                print("Saving sample at {}".format(save_path))
                save_image(sample, save_path, nrow=3, padding=0, normalize=True)
        else:
            print(f"Loading saved gradcam results from {save_path}")
            class_wise_iou = torch.load(save_path, map_location="cpu")
        
        print("Class-wise IOU:")
        print(json.dumps(class_wise_iou, indent=4))

        return save_path


def run_analyses(seed=0, datasets=["colored_MNIST"], show=False, debug=False, ignore_cache=False):
    """Runs multiple analyses on given MNIST variant with given checkpoint."""
    iou_path_og_per_dataset = dict()
    iou_path_cf_per_dataset = dict()

    for dataset in datasets:
        print(f":::::> Running analysis on {dataset} with classifier trained on original dataset")
        mnist_analysis = MNISTAnalysis(
            dataset=dataset,
            weight_path=join(
                REPO_PATH,
                "cgn_framework/mnists/experiments",
                f"classifier_{dataset}_seed_0/weights/ckp_epoch_10.pth",
            ),
            seed=seed,
            ignore_cache=ignore_cache,
        )
        mnist_analysis.visualize_feature(save=True, show=show, num_batches_to_use=5)
        # this step also saves the gradcam images for 10 samples
        iou_path_og = mnist_analysis.perform_gradcam_quantitative(debug=debug)
        iou_path_og_per_dataset[dataset] = iou_path_og

        print(f":::::> Running analysis on {dataset} with classifier trained on CF dataset")
        mnist_analysis = MNISTAnalysis(
            dataset=dataset,
            weight_path=join(
                REPO_PATH,
                "cgn_framework/mnists/experiments",
                f"classifier_{dataset}_counterfactual_seed_0/weights/ckp_epoch_10.pth",
            ),
            seed=seed,
            ignore_cache=ignore_cache,
        )
        mnist_analysis.visualize_feature(save=True, show=show, num_batches_to_use=5)
        # this step also saves the gradcam images for 10 samples
        iou_path_cf = mnist_analysis.perform_gradcam_quantitative(debug=debug)
        iou_path_cf_per_dataset[dataset] = iou_path_cf

        # show qualitative results
        if show:
            print(f"Showing qualitative results for {dataset}")
            images_paths_og = glob(os.path.dirname(iou_path_og) + "/*.png")
            images_paths_cf = glob(os.path.dirname(iou_path_cf) + "/*.png")
            show_gradcam_qualitative(dataset, images_paths_og, images_paths_cf)

    # if show==True, display all figures/results together
    if show:
        plot_iou_for_og_and_cf(iou_path_og_per_dataset, iou_path_cf_per_dataset, show=show)


if __name__ == "__main__":
    run_analyses(
        datasets=["colored_MNIST", "double_colored_MNIST", "wildlife_MNIST"],
        debug=False,
        ignore_cache=False,
    )