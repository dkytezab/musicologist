#!/bin/bash
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=150g
#SBATCH -t 00:45:00
#SBATCH -p batch
#SBATCH -o logs/nsynth_%j.out
#SBATCH -e logs/nsynth_%j.err
#SBATCH -J nsynth

cd data/nsynth

curl --output nsynth-train.jsonwav.tar.gz "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
tar -xzvf nsynth-train.jsonwav.tar.gz
