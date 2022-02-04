from experiment_utils import ImageDirectoryLoader, set_env
set_env()

from inception_score import inception_score, mu_mask, generate_images
import torch
import os

def run_experiments(ignore_cache=False):
    '''
    This is a generator for the inception score and mu mask results used in the ablation study
    '''
    for loss_name in [
        "shape-ablation",
        "text-ablation",
        "bg-ablation",
        "rec-ablation",
    ]:
        cache_file_path = 'imagenet/experiments/loss-ablation/results-' + loss_name + ".pth"
        if os.path.exists(cache_file_path):
            results = torch.load(cache_file_path)
            yield loss_name, results['inception'], results['avg_mask'], results['sd_mask']
        else:
            data_dir = generate_images(f'imagenet/weights/{loss_name}.pth', loss_name)

            images = ImageDirectoryLoader(data_dir + '/ims')
            inception = inception_score(images, splits=2, resize=True)
            avg_mask, sd_mask = mu_mask(data_dir + '/mean_masks.txt')

            torch.save({
                "inception": inception,
                "avg_mask": avg_mask,
                "sd_mask": sd_mask,
            }, cache_file_path)

            yield loss_name, inception, avg_mask, sd_mask
