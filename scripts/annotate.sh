#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=30g
#SBATCH -t 05:00:00
#SBATCH -p batch
#SBATCH -o logs/annotate_%j.out
#SBATCH -e logs/annotate_%j.err
#SBATCH -J annotate

python data/prompts/annotate.py