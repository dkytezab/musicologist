#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=100g
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH -o logs/embeds_%j.out
#SBATCH -e logs/embeds_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J embeddings_generation

module load cuda cudnn
python embeddings/gen_embeds.py
