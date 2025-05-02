#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=40g
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o logs/inference/train_G%j.out
#SBATCH -e logs/inference/train_G%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J diffusion_inference

module load cuda cudnn
python analysis/saes/train_sae.py \
    --data_path data/activations \
    --diff_step 1 \
    --layer 2 \
    --batch_size 4 \
    --latent_dim 256 \
    --verbose
