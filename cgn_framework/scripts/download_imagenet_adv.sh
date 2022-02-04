
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xzf imagenet-a.tar 
mkdir -p ./imagenet/data/in-a/val/
mv imagenet-a/* ./imagenet/data/in-a/val/
rm -rf ./imagenet-a.tar ./imagenet-a/