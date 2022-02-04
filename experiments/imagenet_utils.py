"""Utility functions for ImageNet experiments."""
import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import transforms

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp


def denormalize(x: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes an image."""

    mean = np.array(mean)
    std = np.array(std)
    denormalize_transform = transforms.Normalize(
        mean=-(mean / std),
        std=(1.0 / std),
    )

    return denormalize_transform(x)


class IMModel(torch.nn.Module):

    def __init__(self, base_model, mode):
        super(IMModel, self).__init__()

        self.base_model = base_model
        self.mode = mode
        self.mechanism = f"m_{mode}"
    
    def forward(self, x):
        x = self.base_model.backbone(x)
        x = getattr(self.base_model, self.mechanism)(x)
        return x


class AverageEnsembleModel(torch.nn.Module):

    def __init__(self, base_model):
        super(AverageEnsembleModel, self).__init__()

        self.base_model = base_model
    
    def forward(self, x):
        x = self.base_model.backbone(x)
        x_shape = self.base_model.m_shape(x)
        x_texture = self.base_model.m_texture(x)
        x_bg = self.base_model.m_bg(x)
        x = (x_shape + x_texture + x_bg) / 3.0

        return x


class EnsembleGradCAM:
    """Extension of GradCAM class for our use-case."""

    def __init__(self, ensemble_model, alpha=0.7, target_layer=None, gradcam_method="GradCAM"):
        super(EnsembleGradCAM, self).__init__()

        self.alpha = alpha
        
        self.m_shape = IMModel(base_model=ensemble_model, mode="shape")
        self.gradcam_shape = eval(gradcam_method)(self.m_shape, self.m_shape.base_model.backbone[7][2].conv1)
        
        self.m_texture = IMModel(base_model=ensemble_model, mode="texture")
        self.gradcam_texture = eval(gradcam_method)(self.m_texture, self.m_texture.base_model.backbone[7][2].conv1)

        self.m_bg = IMModel(base_model=ensemble_model, mode="bg")
        self.gradcam_bg = eval(gradcam_method)(self.m_bg, self.m_bg.base_model.backbone[7][2].conv1)
        
        self.m_avg = AverageEnsembleModel(base_model=ensemble_model)
        self.gradcam_avg = eval(gradcam_method)(self.m_avg, self.m_avg.base_model.backbone[7][2].conv1)
    
    def apply_single_gradcam_module(self, image, gradcam_module, prefix):

        image = image.clone()

        # get gradcam mask
        gc_mask, _ = gradcam_module(image.unsqueeze(0))
        heatmap, result = visualize_cam(gc_mask, image)
        gc_mask = gc_mask.squeeze(0)

        # compute custom result: alpha-linear combination of image (denormalized) and heatmap
        denormalized_image = denormalize(image.data.cpu())
        overlap = self.alpha * denormalized_image + (1 - self.alpha) * heatmap.data.cpu()

        # create a grid with original, heatmap and overlapped
        grid_original_heatmap_overlap = make_grid([denormalized_image, heatmap, overlap], nrow=3)
        
        output = {
            "image": denormalized_image,
            f"{prefix}_heatmap": heatmap,
            f"{prefix}_overlap": overlap,
            f"{prefix}_grid_original_heatmap_overlap": grid_original_heatmap_overlap,
        }
        return output

    def apply(self, image: torch.tensor, label: torch.tensor):
        assert len(image.shape) == 3, "Works on a single-image only."
        assert image.shape[0] == 3

        outputs = dict()
        
        mechanisms = ["shape", "texture", "bg", "avg"]
        for i, prefix in enumerate(mechanisms):
            gradcam_module = getattr(self, f"gradcam_{prefix}")
            output = self.apply_single_gradcam_module(image, gradcam_module, prefix)
            outputs.update(output)

        outputs["gt_label"] = label.item()
        for i, prefix in enumerate(mechanisms):
            outputs[f"{prefix}_label"] = getattr(self, f"m_{prefix}")(image.unsqueeze(0)).argmax(1)[0].cpu().item()

        return outputs


def get_imagenet_mini_foldername_to_classname(metadata_file):
    """Returns a dictionary mapping from ImageNet mini-foldername to classname."""
    with open(metadata_file) as f:
        data = f.readlines()

    assert len(data) == 1000
    data = {x.split(" ")[0]:x.split(" ")[2].split("\n")[0] for x in data}
    data = {k:v.replace("_", " ") for k, v in data.items()}

    return data

