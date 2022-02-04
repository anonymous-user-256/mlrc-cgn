## Setup

### Clone the repository

```sh
# clone the repository
git clone git@github.com:anonymous-user-256/mlrc-cgn.git
cd mlrc-cgn.git

# switch to dev branch
git checkout dev
```

### Environment

Depending on whether you have a CPU/GPU machine, install a `conda` environment:
```bash
conda env create --file cgn_framework/environment-gpu.yml 
conda activate cgn-gpu
```

### Download datasets and models

The code to download the datasets and models is in the demo notebook itself.
<!-- 
### Download datasets

The ImageNet-mini dataset needs to be downloaded from Kaggle. Please export your Kaggle credentials using the following command. The key is the Kaggle API key and can be found in your account settings.
```sh
export KAGGLE_USERNAME=<your_username>
export KAGGLE_KEY=<your_key>
```

Or alternatively, you can download your API key `kaggle.json` file and put it here `~/.kaggle/kaggle.json`.

Use the following command to download all required datasets:

```bash
python setup/download_datasets.py
```
This should download datasets for both `mnists` and `imagenet` tasks.

For MNISTs, the folder structure is as follows:
```sh
mnists/data
├── colored_mnist
└── textures
    ├── background
    └── object

4 directories
```

For ImageNet, the folder structure is as follows:
```sh
imagenet/data
├── cue_conflict
├── in-a
├── in-mini
├── in-sketch
├── in-stylized
└── in9

6 directories
```

### Download model weights

Run the following command to download the model weights:

```bash
python setup/download_weights.py
```

This will download the weights for all tasks.

```bash
imagenet/weights/
├── biggan256.pth
├── cgn.pth
├── u2net.pth
└── resnet50_from_scratch_model_best.pth.tar

4 files
``` -->

### Experiments for MNISTs

Please run the `experiments/final-demo.ipynb` notebook to reproduce the results for Table 2.
Further, the same notebook also has code to visualize additional analyses.

### Experiments for ImageNet-mini and OOD

Please run the `experiments/final-demo.ipynb` notebook to reproduce the results for Table 3, 4, 5.
Further, the same notebook also has code to visualize additional analyses.

