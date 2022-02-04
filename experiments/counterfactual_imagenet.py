import torchvision
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import os

from experiment_utils import set_env, dotdict, seed_everything, load_generated_imagenet
set_env()

from cgn_framework.imagenet.generate_data import main as generate_main

def generate_counterfactual(**kwargs):
    args = dotdict(kwargs)
    generate_main(args)

    return args


def main(n_data=6, generate_latex=True):
    seed_everything(19)
    args = generate_counterfactual(mode="best_classes", n_data=n_data, run_name="RUN_NAME", weights_path="imagenet/weights/cgn.pth",
                                   batch_sz=1, truncation=0.5, classes=[0, 0, 0], interp="", interp_cls=-1,
                                   midpoints=6, save_noise=False, save_single=False)

    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    trunc_str = f"{args.run_name}_trunc_{args.truncation}"
    data_path = os.path.join('imagenet', 'data', time_str + trunc_str)

    # Get labels of counterfactual images.
    df_labels = pd.read_csv(os.path.join(data_path, 'labels' + '.csv'), usecols=['shape_cls', 'texture_cls', 'bg_cls'])
    with open(os.path.join('..', 'experiments', 'data', 'imagenet_classes' + '.txt')) as imagenet_file:
        imagenet_classes = {i: line.strip() for i, line in enumerate(imagenet_file)}
    df_labels.replace({'shape_cls': imagenet_classes, 'texture_cls': imagenet_classes, 'bg_cls': imagenet_classes}, inplace=True)

    print(df_labels.T)

    if generate_latex:
        print("\n--= Generated LaTeX table: =--")
        print(df_labels.T.to_latex(header=False))

    # Get counterfactual images
    images_dir = os.path.join(data_path, 'ims')

    # Construct a grid with the generated images
    images = load_generated_imagenet(images_dir, n_data)
    image_grid = make_grid(images, nrow=n_data)

    plt.imshow(image_grid.permute(1, 2, 0))

    # Save the resulting figures accordingly.
    file_name = "qualitative_imagenet"
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "media", "figures", "qualitative")

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    plt.axis("off")

    save_path = os.path.join(directory, file_name + ".pdf")

    print(f"Saving counterfactuals plot to {save_path}..")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()
