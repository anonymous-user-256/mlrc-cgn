"""Gradio demo."""
import re
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
import gradio as gr

import warnings
warnings.filterwarnings("ignore")

from experiment_utils import set_env, REPO_PATH, seed_everything
set_env()

from image_utils import denormalize, show_single_image, show_multiple_images
from cgn_framework.imagenet.dataloader import get_imagenet_dls
from cgn_framework.imagenet.models.classifier_ensemble import InvariantEnsemble
from cgn_framework.imagenet.models import CGN
from experiments.imagenet_utils import (
    EnsembleGradCAM,
    get_imagenet_mini_foldername_to_classname,
)


def display_dummy_image(*inputs: str):
    print(inputs)
    image = np.random.randn(224, 224, 3)
    image = (image - image.min()) / (image.max() - image.min())
    return image


in_mini_folder_to_class = get_imagenet_mini_foldername_to_classname(
    join(REPO_PATH, "cgn_framework/imagenet/data/in-mini/metadata.txt")
)


def display_result_for_mini_imagenet_sample(
        image: torch.Tensor,
        label: int,
        classes: list,
        ensemble_gradcam,
        save=False,
        suffix="sample",
        path="sample_gradcam_label_{}_index_{}.pdf",
        title=None,
        figsize=(14, 4),
    ):
    
    # get gradcam outputs
    outputs = ensemble_gradcam.apply(image, label)
    
    modes = ["shape", "texture", "bg", "avg"]
    images = [outputs["image"]] + [outputs[m + "_overlap"] for m in modes]

    class_name_map = lambda x: in_mini_folder_to_class[classes[x]]
    y_true = class_name_map(outputs['gt_label'])

    subtitles = [f"Original ($y = {y_true}$)"] + \
        ["{} ($\hat y = {}$)".format(m.capitalize(), class_name_map(outputs[m + '_label'])) for m in modes]

    path = path.format(y_true, suffix)
    path = join(REPO_PATH, "experiments", "results", "plots", path)

    fig = show_multiple_images(
        images,
        title=title,
        subtitles=subtitles,
        n_cols=len(images),
        normalized=False,
        figsize=figsize,
        save=save,
        path=path,
        return_figure=True,
        show=False,
    )

    return plt.gcf()


def display_result_for_counterfactual_sample(cgn, ys, classes, clf_transforms, ensemble_gradcam):

    # Generate the output
    # print(":: Generating CF sample ...")
    with torch.no_grad():
        x_gt, mask, premask, foreground, background, bg_mask = cgn(ys=ys)
        x_gen = mask * foreground + (1 - mask) * background
    
    # preprocess the image before send it to the classifier model
    # print(":: Preprocessing CF sample ...")
    image = x_gen[0]
    pil_image = transforms.ToPILImage()((image + 1) * 0.5)
    transformed_image = clf_transforms(pil_image)
    
    # print(":: Generating GradCAM output ...")
    class_name_map = lambda x: in_mini_folder_to_class[classes[x]]
    title = "CF example generated using Shape: {}  \t Texture: {}  \t BG: {}".format(
        class_name_map(ys[0]), class_name_map(ys[1]), class_name_map(ys[2]),
    )
    fig = display_result_for_mini_imagenet_sample(
        image=transformed_image,
        label=torch.tensor(ys[0]),
        classes=classes,
        ensemble_gradcam=ensemble_gradcam,
        suffix=None,
        title=title,
        figsize=(14, 3),
    )

    return plt.gcf()


class CGNGradio:
    """Class that defines the interface for the CGNGradio"""
    def __init__(self, model, cgn, ds_val, df, ensemble_gradcam):
        self.model = model
        self.df = df
        self.cgn = cgn
        self.ds_val = ds_val
        self.ensemble_gradcam = ensemble_gradcam

        self.wordnet_id_to_class_label = get_imagenet_mini_foldername_to_classname(
            join(REPO_PATH, "cgn_framework/imagenet/data/in-mini/metadata.txt")
        )
    
    def configure_input(self):
        """Defines all the input elements."""
        dataset_selector = gr.inputs.Dropdown(
            choices=["ImageNet-Mini", "Counterfactuals"],
            type="value",
            label="Dataset",
        )
        original_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Original label",
        )


        shape_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Shape label",
        )
        texture_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Texture label",
        )
        background_label_selector = gr.inputs.Dropdown(
            choices=sorted(self.wordnet_id_to_class_label.values()),
            type="value",
            label="Background label",
        )
        return [
            dataset_selector,
            original_label_selector,
            shape_label_selector,
            texture_label_selector,
            background_label_selector,
        ]

    def configure_output(self):
        return "plot"

    def display_result(self, *inputs: list):
        dataset, o_label, s_label, t_label, b_label = inputs

        if dataset == "ImageNet-Mini":
            index = self.df[self.df.class_name == o_label].sample(1).sample_index.values[0]
            sample = self.ds_val[index]
            fig = display_result_for_mini_imagenet_sample(
                image=sample["ims"],
                label=sample["labels"],
                ensemble_gradcam=self.ensemble_gradcam,
                classes=self.ds_val.classes,
                suffix=index,
            )
        elif dataset == "Counterfactuals":
            ys = [
                self.df[self.df.class_name == s_label].class_index.unique()[0],
                self.df[self.df.class_name == t_label].class_index.unique()[0],
                self.df[self.df.class_name == b_label].class_index.unique()[0],
            ]
            fig = display_result_for_counterfactual_sample(
                cgn=self.cgn,
                ys=ys,
                classes=self.ds_val.classes,
                clf_transforms=self.ds_val.T_ims,
                ensemble_gradcam=self.ensemble_gradcam,
            )

        return fig

    
    def launch(self, **kwargs):
        inputs = self.configure_input()
        outputs = self.configure_output()
        self.iface = gr.Interface(fn=self.display_result, inputs=inputs, outputs=outputs)
        self.iface.launch(**kwargs)


def init_gradio_module(launch=False, **launch_kwargs):
    seed_everything(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    print("Loading model ...")
    model = InvariantEnsemble("resnet50", pretrained=True)
    ckpt_path = join(
        REPO_PATH,
        "cgn_framework/imagenet/weights/classifier_on_in-mini_model_best.pth",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_state_dict = ckpt["state_dict"]
    ckpt_state_dict = {k.replace("module.", ""):v for k, v in ckpt_state_dict.items()}
    model.load_state_dict(ckpt_state_dict)
    model = model.eval()

    # load data
    print("Loading data ...")
    train_loader, val_loader, train_sampler = get_imagenet_dls(
        "imagenet/data/in-mini", False, 64, 10,
    )
    ds_val = val_loader.dataset
    df = pd.DataFrame(None, columns=["sample_index", "class_index", "class_folder", "class_name"])
    df["sample_index"] = list(range(len(ds_val.labels)))
    df["class_index"] = ds_val.labels.astype(int)
    df["class_folder"] = df["class_index"].apply(lambda x: ds_val.classes[x])
    df["class_name"] = df["class_folder"].replace(in_mini_folder_to_class)

    # load CGN model
    print("Loading CGN model ...")
    cgn = CGN(batch_sz=1, pretrained=False)
    weights = torch.load(join(REPO_PATH, 'cgn_framework/imagenet/weights/cgn.pth'), map_location='cpu')
    cgn.load_state_dict(weights)
    cgn.eval().to(device);

    # define the gradcam module
    print("Defining gradcam module ...")
    ensemble_gradcam = EnsembleGradCAM(ensemble_model=model, gradcam_method="GradCAM")

    cgn_gradio = CGNGradio(model, cgn, ds_val, df, ensemble_gradcam)

    if launch:
        cgn_gradio.launch(**launch_kwargs)

    return cgn_gradio


if __name__ == "__main__":
    init_gradio_module(launch=True, share=True)