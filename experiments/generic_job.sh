#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=SampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm/output_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Activate your environment
source deactivate
source activate gcn-gpu

# Run your code
# echo "Running command: python imagenet_eval_contributions.py"
# python imagenet_eval_contributions.py
# python evaluate_imagenet_sketch.py

# # experiments with ResNet50
# python imagenet_eval_ood_benchmark.py  --classifier resnet50 --ood_dataset in-mini --num_workers 2
# python imagenet_eval_ood_benchmark.py  --classifier resnet50 --ood_dataset in-sketch --num_workers 2
# python imagenet_eval_ood_benchmark.py  --classifier resnet50-from-scratch --ood_dataset in-a --weight_path experiments/weights/resnet50_from_scratch_model_best.pth.tar --num_workers 2
# python imagenet_eval_ood_benchmark.py  --classifier resnet50 --ood_dataset in-stylized --num_workers 2

# # experiments with CGN-ensemble
# python imagenet_eval_ood_benchmark.py  --classifier cgn-ensemble --ood_dataset in-mini --num_workers 2 --weight_path cgn_framework/imagenet/weights/classifier_on_in-mini_model_best.pth
# python imagenet_eval_ood_benchmark.py  --classifier cgn-ensemble --ood_dataset in-sketch --num_workers 2 --weight_path cgn_framework/imagenet/weights/classifier_on_in-mini_model_best.pth
# python imagenet_eval_ood_benchmark.py  --classifier cgn-ensemble --ood_dataset in-stylized --num_workers 2 --weight_path cgn_framework/imagenet/weights/classifier_on_in-mini_model_best.pth
# python imagenet_eval_ood_benchmark.py  --classifier cgn-ensemble --ood_dataset in-a --num_workers 2 --weight_path cgn_framework/imagenet/weights/classifier_on_in-mini_model_best.pth

# python mnist_pipeline.py
# python mnist_analysis.py

python imagenet_pipeline.py