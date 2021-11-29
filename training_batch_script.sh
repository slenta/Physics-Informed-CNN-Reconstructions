#!/bin/bash

#SBATCH -J SimonLentz
#SBATCH -p compute
#SBATCH --account=uo1075
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --mem=32G

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity run --bind /work/uo1075/u301617/ /work/uo1075/u301617/Master-Arbeit/pytorch_gpu_new.sif \ 
 python /work/uo1075/u301617/Master-Arbeit/train.py \\
 

