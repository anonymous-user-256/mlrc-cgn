# python experiments/quant_metric_for_gradcam.py \
#     --dataset double_colored_MNIST \
#     --weight_path mnists/experiments/classifier_double_colored_MNIST/weights/ckp_epoch_10.pth

python experiments/quant_metric_for_gradcam.py \
    --dataset double_colored_MNIST \
    --weight_path mnists/experiments/classifier_double_colored_MNIST_counterfactual/weights/ckp_epoch_10.pth

# python experiments/quant_metric_for_gradcam.py \
#     --dataset wildlife_MNIST \
#     --weight_path mnists/experiments/classifier_wildlife_MNIST/weights/ckp_epoch_10.pth

# python experiments/quant_metric_for_gradcam.py \
#     --dataset wildlife_MNIST \
#     --weight_path mnists/experiments/classifier_wildlife_MNIST_counterfactual/weights/ckp_epoch_10.pth