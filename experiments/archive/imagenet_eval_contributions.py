"""Evaluates the contributions of the different heads in the invariant classifier."""
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
import torch

from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision import transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from experiments.image_utils import denormalize, show_single_image
from cgn_framework.imagenet.dataloader import get_imagenet_dls
from cgn_framework.imagenet.models.classifier_ensemble import InvariantEnsemble
from cgn_framework.imagenet.models import CGN
from cgn_framework.imagenet.generate_data import sample_classes

# set plt params to get LaTeX-like figures
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def plot_individual_head_logprobs(
        y_shape_all, y_texture_all, y_bg_all, show=False,
        save_path=None, save=False,
    ):
    """Plots the logprobs of the individual heads against each other."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].scatter(y_texture_all, y_shape_all)
    corr = np.corrcoef(y_texture_all, y_shape_all)[0, 1]
    ax[0].set_title(f"Texture vs Shape ($P = {corr:.3f}$)")
    ax[0].grid()

    ax[1].scatter(y_bg_all, y_shape_all)
    corr = np.corrcoef(y_bg_all, y_shape_all)[0, 1]
    ax[1].set_title(f"Background vs Shape ($P = {corr:.3f}$)")
    ax[1].grid()

    ax[2].scatter(y_bg_all, y_texture_all)
    corr = np.corrcoef(y_bg_all, y_texture_all)[0, 1]
    ax[2].set_title(f"Background vs Texture ($P = {corr:.3f}$)")
    ax[2].grid()

    if save:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()


if __name__ == "__main__":
    seed_everything(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## load original dataset
    train_loader, val_loader, train_sampler = get_imagenet_dls(
        "imagenet/data/in-mini", False, 32, 4,
    )
    ds_val = val_loader.dataset

    ## generate new CF samples
    # load imagenet classes
    imagenet_classes_file = join(
        REPO_PATH,
        "cgn_framework/imagenet/data/in-mini/imagenet_classes.txt",
    )
    if not os.path.exists(imagenet_classes_file):
        url = "wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        os.system(f"wget {url} -O {imagenet_classes_file}")
    with open(imagenet_classes_file, "r") as f:
        imagenet_categories = [s.strip() for s in f.readlines()]
    # load CGN model
    cgn = CGN(batch_sz=1, pretrained=False)
    # load weights and push to GPU
    weights = torch.load(
        join(REPO_PATH, 'cgn_framework/imagenet/weights/cgn.pth'),
        map_location='cpu',
    )
    cgn.load_state_dict(weights)
    cgn.eval().to(device)

    ## load the classifier model
    model = InvariantEnsemble("resnet50", pretrained=True)
    # load weights from a checkpoint
    ckpt_path = join(
        REPO_PATH, 
        "cgn_framework/imagenet/experiments/classifier_2022_01_19_15_36_sample_run/model_best.pth",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = ckpt["state_dict"]
    ckpt_state_dict = {k.replace("module.", ""):v for k, v in ckpt_state_dict.items()}
    model.load_state_dict(ckpt_state_dict)
    model = model.eval().to(device)

    ## generate results for original data
    iterator = tqdm(
        val_loader,
        desc="Evaluating on validation set",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )

    y_shape_all = []
    y_texture_all = []
    y_bg_all = []
    with torch.no_grad():
        for batch in iterator:

            ims = batch["ims"].to(device)
            labels = batch["labels"].to(device)

            out = model(ims)

            y_shape = out["shape_preds"][np.arange(0, len(labels), 1), labels]
            y_texture = out["texture_preds"][np.arange(0, len(labels), 1), labels]
            y_bg = out["bg_preds"][np.arange(0, len(labels), 1), labels]
            
            y_shape_all.append(y_shape.data.cpu())
            y_texture_all.append(y_texture.data.cpu())
            y_bg_all.append(y_bg.data.cpu())

    y_shape_all = torch.cat(y_shape_all)
    y_texture_all = torch.cat(y_texture_all)
    y_bg_all = torch.cat(y_bg_all)

    plot_individual_head_logprobs(
        y_shape_all,
        y_texture_all,
        y_bg_all,
        save=True,
        save_path=join(REPO_PATH, "experiments/results/plots/joint_clf_heads_logprobs_og.pdf"),
    )

    ## generate results for CF samples
    # Get the input classes
    num_samples = len(ds_val) // 5
    ys_all = []

    y_shape_all = []
    y_texture_all = []
    y_bg_all = []
    for i in tqdm(range(num_samples), desc="Evaluating on CF examples"):
        ys = sample_classes('best_classes')
        labels = [ys[0]]

        # Generate the output
        with torch.no_grad():
            # get generated image (in [-1, 1])
            x_gt, mask, premask, foreground, background, bg_mask = cgn(ys=ys)
            x_gen = mask * foreground + (1 - mask) * background
            
            # forward pass the generated image
            image = x_gen[0]
            pil_image = transforms.ToPILImage()((image + 1) * 0.5)
            transformed_image = ds_val.T_ims(pil_image)
            transformed_image = transformed_image.to(device)
            
            # get classifier outputs
            out = model(transformed_image.unsqueeze(0))

            y_shape = out["shape_preds"][np.arange(0, len(labels), 1), labels]
            y_texture = out["texture_preds"][np.arange(0, len(labels), 1), labels]
            y_bg = out["bg_preds"][np.arange(0, len(labels), 1), labels]

            y_shape_all.append(y_shape.data.cpu())
            y_texture_all.append(y_texture.data.cpu())
            y_bg_all.append(y_bg.data.cpu())

    y_shape_all = torch.cat(y_shape_all)
    y_texture_all = torch.cat(y_texture_all)
    y_bg_all = torch.cat(y_bg_all)

    plot_individual_head_logprobs(
        y_shape_all,
        y_texture_all,
        y_bg_all,
        save=True,
        save_path=join(REPO_PATH, "experiments/results/plots/joint_clf_heads_logprobs_cf.pdf"),
    )
