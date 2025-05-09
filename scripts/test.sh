#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH -o logs/test_interp_%j.out
#SBATCH -e logs/test_interp_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J test_interp

conda activate musicologist
python interp/main.py