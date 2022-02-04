"""Script to download all required datasets at apt location."""
import os
from os.path import join, exists, isdir
from glob import glob
import sys
import shutil
import subprocess
import tarfile
import zipfile

import repackage
repackage.up()

from experiments.experiment_utils import set_env, REPO_PATH
set_env()


def check_existing_datasets():
    root = join(REPO_PATH, "cgn_framework/mnists/data")
    colored_mnist = join(root, "colored_mnist")
    obj_texture_files = glob(join(root, "textures/object/*.jpg"))
    bg_texture_files = glob(join(root, "textures/background/*.jpg"))
    
    root = join(REPO_PATH, "cgn_framework/imagenet/data")
    cue_conflict_dir = join(root, "cue_conflict")

    check = isdir(colored_mnist) and len(obj_texture_files) == 10 and \
        isdir(cue_conflict_dir) and len(bg_texture_files) == 10
    
    return check


def check_downloaded_datasets():
    # check MNISTs
    root = join(REPO_PATH, "cgn_framework/mnists/data")
    
    colored_mnist = join(root, "colored_mnist")
    npy_files = glob(join(colored_mnist, "*.npy"))
    assert len(npy_files) == 7, \
        f"Expected 7 npy files in {colored_mnist}. "\
        f"Probably the files for colored_mnist/ are not downloaded."
    
    obj_texture_files = glob(join(root, "textures/object/*.jpg"))
    assert len(obj_texture_files) == 10, \
        f"Expected 10 object texture files in {root}/textures/objects. "\
        f"Probably the files for textures/objects/ are not downloaded."
    bg_texture_files = glob(join(root, "textures/background/*.jpg"))
    assert len(bg_texture_files) == 10, \
        f"Expected 10 background texture files in {root}/textures/background. "\
        f"Probably the files for textures/background/ are not downloaded."
    
    # check ImageNet files
    root = join(REPO_PATH, "cgn_framework/imagenet/data")

    cue_conflict_dir = join(root, "cue_conflict")
    classes = glob(join(cue_conflict_dir, "*"))
    assert len(classes) == 16, \
        "Cue conflict dataset not downloaded properly."
    
    in9_dir = join(root, "in9")
    assert len(glob(join(in9_dir, "*"))) == 9, \
        f"Expected 9 folders in {in9_dir}. "\
        f"Probably the files for imagenet/data/in9/ are not downloaded."
    
    return True
    

if __name__ == "__main__":
    # download colored MNIST, IN-9, Cue-conflict datasets (following their script)
    print("::: Downloading datasets: colored MNIST | IN-9 | Cue-conflict ...")
    if check_existing_datasets():
        print("- All datasets required for reproducibility are already downloaded.")
    else:
        subprocess.call("bash scripts/download_data.sh", shell=True)
        check = check_downloaded_datasets()
    print("")

    # download other datasets
    
    # download OOD datasets
    print("::: Downloading OOD datasets ...")

    print("1. Downloading IN-a ...")
    in_a_dir = join(REPO_PATH, "cgn_framework/imagenet/data/in-a/val/")
    if not isdir(in_a_dir):
        subprocess.call("bash ./scripts/download_imagenet_adv.sh", shell=True)
    else:
        print("- IN-a is already downloaded. \n")
    
    print("2. Downloading IN-sketch ...")
    in_sketch_dir = join(REPO_PATH, "cgn_framework/imagenet/data/in-sketch/val/")
    if not isdir(in_sketch_dir):
        subprocess.call("bash ./scripts/download_imagenet_sketch.sh", shell=True)
    else:
        print("- IN-sketch is already downloaded. \n")

    # download IN-mini
    print("3. Downloading IN-stylized ...")
    in_stylized_dir = join(REPO_PATH, "cgn_framework/imagenet/data/in-stylized/val/")
    if not isdir(in_stylized_dir):
        subprocess.call("bash ./scripts/download_imagenet_stylized.sh", shell=True)
    else:
        print("- IN-stylized is already downloaded. \n")

    print("::: Downloading IN-mini ...")
    data_dir = join(REPO_PATH, "cgn_framework/imagenet/data/in-mini/train")
    if not isdir(data_dir):
        zip_file = join(REPO_PATH, "cgn_framework/imagenet/data/imagenetmini-1000.zip")
        if not exists(zip_file):
            subprocess.call("kaggle datasets download -d ifigotin/imagenetmini-1000", shell=True)
            subprocess.call("mv imagenetmini-1000.zip {}".format(zip_file), shell=True)

        print(f"- Unzipping {zip_file}...")
    
        unzip_dir = join(REPO_PATH, "cgn_framework/imagenet/data/imagenet-mini/")
        if not isdir(unzip_dir):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(zip_file.replace("imagenetmini-1000.zip", ""))
        subprocess.call(f"mv {unzip_dir}/* {unzip_dir.replace('imagenet-mini', 'in-mini')}", shell=True)
        subprocess.call(f"rm -rf {unzip_dir} {zip_file}", shell=True)
        print("- Done \n")
    else:
        print("- IN-mini is already downloaded.")



