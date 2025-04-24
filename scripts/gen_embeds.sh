#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=100g
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o logs/inference/train_G%j.out
#SBATCH -e logs/inference/train_G%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J embeddings_generation

module load cuda cudnn
python embeddings/gen_embeds.py
