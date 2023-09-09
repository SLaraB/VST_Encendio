#!/bin/bash

# Partition and GPU
#SBATCH -p cpu
##SBATCH --gres=gpu:A100:1

# Identification
#SBATCH -J CPUEval
#SBATCH -o CPU_Evaluation.log

pwd
date
echo "Evaluation CPU 306 images"
srun --container-name=jax --container-workdir=${PWD} python3 densenet.py
srun --container-name=jax --container-workdir=${PWD} python3 inception.py
srun --container-name=jax --container-workdir=${PWD} python3 resnet.py
srun --container-name=jax --container-workdir=${PWD} python3 vgg16.py
srun --container-name=jax --container-workdir=${PWD} python3 vgg19.py
date
echo "Evaluation CPU Completed"