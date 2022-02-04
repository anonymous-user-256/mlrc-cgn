"""
Computes a quantitative metric for Grad-CAM obtained heatmaps.
Basically, computed an IoU between the heatmap and the ground truth (on MNISTs).
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


def main(args):
    seed_everything(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and its weights from a checkpoint
    model = CNN()

    # load checkpoint
    print("Loading model weights from checkpoint: {}".format(args.weight_path))
    ckpt = torch.load(os.path.join(REPO_PATH, "cgn_framework", args.weight_path), map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.eval()
    model = model.to(device)

    # load the test dataloader to evaluate GradCAM on
    print("Loading dataset: {}".format(args.dataset))
    dl_train, dl_test = get_tensor_dataloaders(args.dataset, 64)
    ds_test = dl_test.dataset

    # load original MNIST to obtain binary maps
    T = tv_transforms.Compose([
        tv_transforms.Resize((32, 32), Image.NEAREST),
        tv_transforms.ToTensor(),
    ])
    original = tv_datasets.MNIST(
        root="./mnists/data/MNIST/", download=True, train=False, transform=T,
    )

    # define the target layer to be used
    target_layer = model.model[8]
    gradcam = GradCAM(model, target_layer)

    # for metric computation
    jaccard = JaccardIndex(num_classes=2)
    jaccard = jaccard.to(device)

    # apply GradCAM on the test set
    iterator = tqdm(
        range(len(ds_test)),
        colour="red",
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        desc="Computing GradCAM",
        disable=args.disable_tqdm,
    )
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

    return class_wise_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=TENSOR_DATASETS,
                        help='Provide dataset name.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='path to the classifier checkpoint')
    parser.add_argument('--disable_tqdm', action='store_true',
                        help='disable tqdm progress bar')

    args = parser.parse_args()

    print("::::: Configs :::::")
    print(args)

    print("::::: Running :::::")
    results = main(args)

    # save results
    args.model = args.weight_path.split("/")[-3]
    save_path = os.path.join(
        REPO_PATH,
        "experiments",
        "results",
        f"{args.dataset}_{args.model}_seed_{args.seed}_gradcam_iou.pth",
    )
    torch.save(results, save_path)

    print("::::: Results :::::")
    print(json.dumps(str(results), indent=4))

