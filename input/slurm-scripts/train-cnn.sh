#!/bin/bash

#SBATCH -J crai-training
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --output=outputs/train-cnn.o%j 

module load python3/2022.01-gcc-11.2.0
cd /work/uo1075/u301617/Master_Arbeit/code-goratz/

source activate crai
python -m climatereconstructionai.train --load-from-file ./input/levante/train-cnn.txt --max-iter 300000 --resume-iter 275000
