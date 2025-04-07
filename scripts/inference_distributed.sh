#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=30g
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o logs/inference/train_G%j.out
#SBATCH -e logs/inference/train_G%j.err
#SBATCH --gres=gpu:2
#SBATCH --constraint=ampere
#SBATCH -J dist_diffusion_inference

module load cuda cudnn
torchrun --nproc_per_node=2 diffusion/generate_distributed.py
