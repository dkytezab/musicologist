#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=30g
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o logs/inference/train_G%j.out
#SBATCH -e logs/inference/train_G%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J diffusion_inference

module load cuda cudnn
python pipeline/diffusion/generate.py
