import readline

from experiment_utils import set_env, dotdict
if __name__ == "__main__":
    set_env()

from inception_score_pytorch.inception_score import inception_score
from experiment_utils import ImageDirectoryLoader

from torch.utils.data import Dataset, DataLoader, TensorDataset

import argparse
import torch
import os


def generate_images(weights_path, run_name):
    '''
    Use the CGN to generate images on which the inception score and mu_mask will be calculated.
    '''
    from cgn_framework.imagenet.generate_data import main as generate_main

    args = dotdict({
        "mode": "random_same",
        "weights_path": weights_path,
        "ignore_time_in_filename": True,
        "n_data": 2000,
        "run_name": run_name,
        "truncation": 0.5,
        "batch_sz": 1,
    })

    final_dir_name = f"imagenet/data/{run_name}_trunc_{args.truncation}"

    if os.path.exists(final_dir_name):
        print("Generated data already exists. It will be used instead of regenerated.")
    else:
        generate_main(args)

    return final_dir_name

def mu_mask(file_path):
    '''
    Reads the mu_mask values from the text file at location `file_path` (one float per line).
    Then calculates their mean and standard deviation.
    '''
    with open(file_path, 'r') as f:
        mus = f.readlines()
        mus_count = len(mus)

    mus = [float(mu[:-1]) for mu in mus]
    avg_mask = sum(mus) / mus_count
    sds_mask = ((mu - avg_mask) ** 2 for mu in mus)
    sd_mask = (sum(sds_mask) / mus_count) ** 0.5
    return avg_mask, sd_mask

def main(args):
    if args.data_dir is None:
        assert args.run_name is not None and args.weights_path is not None, "if the data_dir argument is not supplied, supply run_name and weight_path"
        data_dir = generate_images(args.weights_path, args.run_name)
    else:
        data_dir = args.data_dir

    images = ImageDirectoryLoader(data_dir + '/ims')
    inception = inception_score(images, splits=2, resize=True)

    print('inception score =', inception)

    print('mu_mask, sd_mask =', mu_mask(data_dir + '/mean_masks.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Folder to load the images from. If the images '
                        'do not exist in this folder, the program genetates '
                        'new images and stores those in this folder.')
    parser.add_argument('--weights_path', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()

    main(args)
