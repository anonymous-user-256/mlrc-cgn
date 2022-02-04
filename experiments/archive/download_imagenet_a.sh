
REPO_PATH="$( dirname $(dirname "$(realpath "$0")"))"
echo "Downloading ImageNet-Sketch at "$REPO_PATH/cgn_framework/imagenet/data/sketch

wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvz imagenet-a.tar 
mv imagenet-a/* $REPO_PATH/cgn_framework/imagenet/data/in-a/val/
rm -rf imagenet-a.tar