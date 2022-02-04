# Downloads the ImageNet-mini dataset and stores it in the specified directory.
# wget "https://www.kaggle.com/ifigotin/imagenetmini-1000/download" -O imagenet_mini.zip

# Download the zip file from https://www.kaggle.com/ifigotin/imagenetmini-1000
# You need a Kaggle login to download the dataset.

# add it to the repository folder as cgn_framework/imagenet_mini.zip

# unzip the file
unzip imagenet_mini.zip -d imagenet_mini

# move to correct folder
rsync -avzP imagenet_mini/ imagenet/data/in-mini/

# delete the zip file
rm -rf imagenet_mini.zip