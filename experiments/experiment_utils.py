import os, sys
from os.path import join, dirname, abspath
import random
import numpy as np
import torch


REPO_PATH = dirname(dirname(abspath(__file__)))

import itertools

from torch.utils.data import Dataset
import torchvision

def set_env():
    """This function does two things:
    1. It allows the code in cgn_framework to be imported from the experiments folder.
    2. It changes the current working directory to cgn_framework/. This is necessary,
    because the code present in this directory expects to be called from here."""
    sys.path.insert(0, REPO_PATH)
    sys.path.insert(1, join(REPO_PATH, 'cgn_framework'))
    os.chdir(join(REPO_PATH, "cgn_framework"))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_everything(seed):
    """Fixes random seed for all modules in the framework."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_generated_imagenet(images_dir, images_count=None):
    """
    Loads the images in the specified directory eagerly
    """
    # Get the locations of the generated images
    image_range = range(images_count) if images_count is not None else itertools.count()
    image_paths = (images_dir + "/" + path for path, _ in zip(os.listdir(images_dir), image_range))

    return [torchvision.io.read_image(path) for path in image_paths]


class ImageDirectoryLoader(Dataset):
    """
    Loads the images in the specified directory
    """
    def __init__(self, images_dir):
        self.image_paths = list((images_dir + "/" + path for path in os.listdir(images_dir)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        return torchvision.io.read_image(self.image_paths[i]) / 127.5 - 1
