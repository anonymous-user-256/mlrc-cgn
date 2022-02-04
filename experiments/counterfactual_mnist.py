"""
Authors: Anonymous

counterfactual_mnist.py

This program generates counterfactual images for the MNIST dataset
and plots them. Datasets include colored MNIST, double-coloured MNIST
and wilflife MNIST.
"""


import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import os

from experiment_utils import set_env, seed_everything
set_env()

from cgn_framework.mnists.generate_data import generate_cf_dataset, generate_dataset, get_dataloaders
from cgn_framework.mnists.train_cgn import CGN


DATASETS = {"colored_MNIST": "cgn_colored_MNIST",
            "double_colored_MNIST": "cgn_double_colored_MNIST",
            "wildlife_MNIST": "cgn_wildlife_MNIST"
            }


def generate_counterfactual(dataset_size=100000, no_cfs=10):
    """Generate the counterfactual images for the 3 datasets."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset, weight_folder in DATASETS.items():
        # First create the real images.
        dl_train, dl_test = get_dataloaders(dataset, batch_size=1000, workers=8)

        if os.path.exists(f"mnists/data/{dataset}_train.pth"):
            print(f"'{dataset}_train.pth' already exist, skipping..")
        else:
            generate_dataset(dl=dl_train, path=dataset + '_train.pth')

        if os.path.exists(f"mnists/data/{dataset}_test.pth"):
            print(f"'{dataset}_test.pth' already exist, skipping..")
        else:
            generate_dataset(dl=dl_test, path=dataset + '_test.pth')

        # Next, generate the counterfactual images.
        weight_path = f"mnists/experiments/{weight_folder}/weights/ckp.pth"

        if os.path.exists(f"mnists/data/{dataset}_counterfactual_samples.pth"):
            print(f"'{dataset}_counterfactual_samples.pth' already exist, skipping..")
        else:
            cgn = CGN()
            cgn.load_state_dict(torch.load(weight_path, 'cpu'))
            cgn.to(device).eval()
            print(f"Generating the counterfactual {dataset} of size {dataset_size}")
            generate_cf_dataset(cgn=cgn, path=dataset + '_counterfactual_samples.pth',
                                dataset_size=dataset_size, no_cfs=no_cfs,
                                device=device)


def main(dataset_size=100000, no_cfs=10):
    seed_everything(13)
    generate_counterfactual(dataset_size=dataset_size, no_cfs=no_cfs)

    # Plot the real and counterfactual iamges (i.e., Figure 3 of the paper).
    for dataset, weight_folder in DATASETS.items():
        visualise_generated_images(f"mnists/data/{dataset}_train.pth", title=f"{dataset}: Original")
        visualise_generated_images(f"mnists/data/{dataset}_counterfactual_samples.pth", title=f"{dataset}: Counterfactual")


def visualise_generated_images(path, title=None):
    images, labels = torch.load(path)

    # Transform the image range [-1, 1] to the range [0, 1]
    images = images * 0.5 + 0.5

    digits_count = 4
    examples_per_digit = 3

    shown_examples = torch.empty((digits_count * examples_per_digit,) + images.shape[1:])

    # Pick example images, with the first row containing 0's, the second containing 1's, etc.
    for digit in range(digits_count):
        i = digit * examples_per_digit
        shown_examples[i:i + examples_per_digit] = images[labels == digit][:examples_per_digit]

    # Rows and columns are swapped because matplotlib has different axes
    grid = make_grid(shown_examples, nrow=examples_per_digit)

    #TODO: Explain the permute.
    plt.imshow(grid.permute((1, 2, 0)))
    plt.title(title)

    # Save the resulting figures accordingly.
    file_name = "qualitative_" + path.split("/")[-1].split(".")[0]
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "media", "figures", "qualitative")

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    plt.axis("off")
    plt.savefig(os.path.join(directory, file_name + ".pdf"), bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
