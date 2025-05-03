#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=30g
#SBATCH -t 00:30:00
#SBATCH -p gpu
#SBATCH -o logs/inference_distrib/train_G%j.out
#SBATCH -e logs/inference_distrib/train_G%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH -J diffusion_inference_distrib
#SBATCH --array=0-15            # launch tasks 0,1,2,3 → num_jobs=4


module load cuda cudnn
srun python diffusion/gen_distrib.py \
     --job-index $SLURM_ARRAY_TASK_ID \
     --num-jobs  $SLURM_ARRAY_TASK_COUNT
