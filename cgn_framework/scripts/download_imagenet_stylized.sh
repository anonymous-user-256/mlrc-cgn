git clone git@github.com:rgeirhos/texture-vs-shape.git
mkdir -p ./imagenet/data/in-stylized/val/
rsync -azP texture-vs-shape/stimuli/style-transfer-preprocessed-512/* ./imagenet/data/in-stylized/val/
rm -rf texture-vs-shape/