"""Helper functions for GradCAM analysis.

Referece: https://linuxtut.com/en/082f71b96b9aca0d5df5/
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torchvision.datasets as tv_datasets
from torchvision import transforms as tv_transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import JaccardIndex

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from cgn_framework.mnists.models.classifier import CNN
from cgn_framework.mnists.train_cgn import save
from cgn_framework.mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS



def compute_iou_between_gt_and_gradcam_for_mnist(
        dataset, weight_path, seed=0, disable_tqdm=False, debug=False, return_samples=False,
    ):
    """
    Compute IoU between ground truth and GradCAM for MNIST.

    Args:
        dataset (str): dataset name
        weight_path (str): absolute path to the checkpoint file
        seed (int): random seed
        disable_tqdm (bool): whether to disable tqdm progress bar
        debug (bool): whether to print debug messages
        return_samples (bool): whether to return samples of the gradcam
    """
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and its weights from a checkpoint
    model = CNN()

    # load checkpoint
    print("Loading model weights from checkpoint: {}".format(weight_path))
    ckpt = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.eval()
    model = model.to(device)

    # load the test dataloader to evaluate GradCAM on
    print("Loading dataset: {}".format(dataset))
    dl_train, dl_test = get_tensor_dataloaders(dataset, 64)
    ds_test = dl_test.dataset

    # load original MNIST to obtain binary maps
    T = tv_transforms.Compose([
        tv_transforms.Resize((32, 32), Image.NEAREST),
        tv_transforms.ToTensor(),
    ])
    original_data_root = os.path.join(REPO_PATH, "cgn_framework/mnists/data/MNIST/")
    original = tv_datasets.MNIST(
        root=original_data_root, download=True, train=False, transform=T,
    )

    # define the target layer to be used
    target_layer = model.model[8]
    gradcam = GradCAM(model, target_layer)

    # for metric computation
    jaccard = JaccardIndex(num_classes=2)
    jaccard = jaccard.to(device)

    # apply GradCAM on the test set
    num_samples = len(ds_test) if not debug else 10
    np.random.seed(0)
    sample_indices = np.random.choice(num_samples, 10, replace=False)
    iterator = tqdm(
        range(num_samples),
        colour="red",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        desc="Computing GradCAM",
        disable=disable_tqdm,
    )
    samples = []
    iou_values = []
    labels = []
    show = False
    for i in iterator:

        # get image
        image, label = ds_test[i]
        image = image.to(device)

        # get gradcam mask
        gc_mask, _ = gradcam(image.unsqueeze(0))
        heatmap, result = visualize_cam(gc_mask, image)
        gc_mask = gc_mask.squeeze(0).to(device)

        # get coresponding GT mask from original dataset
        gt_mask = original[i][0]
        gt_mask = gt_mask.to(device)
        gt_mask_binary = gt_mask > 0.5

        iou = jaccard(gc_mask, gt_mask_binary)
        iou_values.append(iou.cpu().item())
        labels.append(label)

        if return_samples and i in sample_indices:
            samples.append([image.data.cpu(), heatmap, result])
        
        if show:
            grid = make_grid([gt_mask, gc_mask], padding=0)
            plt.title(f"IoU: {iou:.4f}", fontsize=18)
            plt.imshow(grid.permute((1, 2, 0)))
            plt.show()

    df = pd.DataFrame(None, columns=["iou", "label"])
    df["iou"] = iou_values
    df["label"] = torch.stack(labels).numpy()

    class_wise_results = dict(df.groupby("label")["iou"].mean())
    class_wise_results["overall_mean"] = np.mean(iou_values)
    class_wise_results["overall_std"] = np.std(iou_values)

    if return_samples:
        return class_wise_results, sample_indices, samples
    else:
        return class_wise_results
