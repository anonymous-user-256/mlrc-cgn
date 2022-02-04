# Downloads the ImageNet Sketch dataset.
REPO_PATH="$( dirname $(dirname "$(realpath "$0")"))"
echo "Downloading ImageNet-Sketch at "$REPO_PATH/cgn_framework/imagenet/data/sketch

gdown "https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA" -O $REPO_PATH/cgn_framework/imagenet/data/
unzip $REPO_PATH/cgn_framework/imagenet/data/ImageNet-Sketch.zip -d $REPO_PATH/cgn_framework/imagenet/data/
rm -rf $REPO_PATH/cgn_framework/imagenet/data/ImageNet-Sketch.zip
mkdir -p $REPO_PATH/cgn_framework/imagenet/data/in-sketch/val/
mv $REPO_PATH/cgn_framework/imagenet/data/in-sketch/* $REPO_PATH/cgn_framework/imagenet/data/in-sketch/val/