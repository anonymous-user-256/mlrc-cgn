## Experiments to reproduce Table 2


### Introduction

This experiment is a reproduction of the results of Table 2 in the paper.
Results are reported in sheet `Table 2` in [this spreadsheet](https://docs.google.com/spreadsheets/d/19jrl_rQnsTDQahcWdqJPeW1T5FzWay7OE_rt_e96zEM/edit?usp=sharing).


#### Setup datasets
1. Download the datasets by following their commands to download. (Make sure you have `gdown`)

Make all scripts executable: ```chmod +x  scripts/*```. Then, download the datasets (colored MNIST, Cue-Conflict, IN-9) and the pre-trained weights (CGN, U2-Net). Comment out the ones you don't need.

```Shell
cd cgn_framework/
./scripts/download_data.sh
./scripts/download_weights.sh
```

#### Train on original datasets

1. Create tensor datasets for each of those
   ```bash
   cd cgn_framework/
   python mnists/generate_data.py --dataset colored_MNIST
   python mnists/generate_data.py --dataset double_colored_MNIST
   python mnists/generate_data.py --dataset wildlife_MNIST
   ```
2. Train the classifier (replace dataset with the dataset you want to train on)
   ```bash
   python mnists/train_classifier.py --dataset colored_MNIST
   ```

#### Training on `Original + CGN`
1. Generate CF data for each dataset
2. Train the classifier

#### Training on `Original + GAN`
1. Currently, code for GAN is not available (architecture is also not known). We added code for Generator(s) (`mnists/models/generator.py`) and for training GAN (`mnists/train_gan.py`).
2. To train GAN, you need to run the following command:
   ```bash
   python mnists/train_gan.py --dataset colored_MNIST
   ```
   For dataset `wildlife_MNIST`, you need to run the following command:
   ```bash
   python mnists/train_gan.py --dataset wildlife_MNIST
   ```
   For dataset `double_colored_MNIST`, you need to run the following command:
   ```bash
   python mnists/train_gan.py --dataset double_colored_MNIST
   ```
3. To generate data using a GAN, you can use the following commands. Note that we have added pretrained GAN checkpoints at relevant locations.
   ```bash
   python mnists/generate_data.py --dataset colored_MNIST --weight_path mnists/experiments/gan_colored_MNIST/weights/ckp.pth
   python mnists/generate_data.py --dataset double_colored_MNIST --weight_path mnists/experiments/gan_double_colored_MNIST/weights/ckp.pth
   python mnists/generate_data.py --dataset wildlife_MNIST --weight_path mnis/experiments/gan_wildlife_MNIST/weights/ckp.pth
   ```
4. Train the classifier (replace dataset with the dataset you want to train on)
   ```bash
   python mnists/train_classifier.py --dataset colored_MNIST_gan
   ```


#### Training `IRM` (invariance via model)
1. Check out code [here](https://github.com/facebookresearch/InvariantRiskMinimization). See if we can use it out-of-the-box
2. Code for colored MNIST exists [here](https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py)

#### Training `LNTL` (invariance via model) [code here](https://github.com/feidfoe/learning-not-to-learn)
