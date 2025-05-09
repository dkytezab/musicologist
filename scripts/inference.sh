#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=30g
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o logs/inference_%j.out
#SBATCH -e logs/inference_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J diffusion_inference

module load cuda cudnn
python diffusion/gen_audio.py
