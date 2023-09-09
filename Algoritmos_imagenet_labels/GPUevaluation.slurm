#!/bin/bash

# Partition and GPU
#SBATCH -p gpu
#SBATCH --gres=gpu:A100:1

# Identification
#SBATCH -J GPUEval
#SBATCH -o GPU_Evaluation.log

pwd
date
echo "Evaluation GPU 306 images"
srun --container-name=jax --gres=gpu:A100:1 --container-workdir=${PWD} python3 densenet.py
srun --container-name=jax --gres=gpu:A100:1 --container-workdir=${PWD} python3 inception.py
srun --container-name=jax --gres=gpu:A100:1 --container-workdir=${PWD} python3 resnet.py
srun --container-name=jax --gres=gpu:A100:1 --container-workdir=${PWD} python3 vgg16.py
srun --container-name=jax --gres=gpu:A100:1 --container-workdir=${PWD} python3 vgg19.py
date
echo "Evaluation GPU Completed"