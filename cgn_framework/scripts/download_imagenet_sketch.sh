# Downloads the ImageNet Sketch dataset.

gdown "https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA"
unzip -qq ImageNet-Sketch.zip

mkdir -p ./imagenet/data/in-sketch/val/
mv ./sketch/* ./imagenet/data/in-sketch/val/
rm -rf ImageNet-Sketch.zip sketch/
