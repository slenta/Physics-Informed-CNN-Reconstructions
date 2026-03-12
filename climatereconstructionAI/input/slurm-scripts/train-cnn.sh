#!/bin/bash

#SBATCH -J crai-training
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --output=/work/uo1075/u241308/ML_downscaling/code/climatereconstructionAI/input/slurm-scripts/train_slurm_output/train-cnn.o%j 

#module load python3/2023.01-gcc-11.2.0
module load python3
source activate crai_downscalling
cd /work/uo1075/u241308/ML_downscaling/code/climatereconstructionAI/

python -m climatereconstructionai.train --load-from-file ./input/levante/train-crai-downscalling.txt --max-iter 700000
