#!/bin/bash

#SBATCH -J SimonLentz
#SBATCH -p gpu
#SBATCH --account=uo1075
#SBATCH -n 1
#SBATCH --nodelist=mg206
#SBATCH --time=8:00:00
#SBATCH --mem=32G

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity exec --bind /work/uo1075/u301617/ /work/uo1075/u301617/Master-Arbeit/pytorch_gpu_new.sif \
 python train.py
 

