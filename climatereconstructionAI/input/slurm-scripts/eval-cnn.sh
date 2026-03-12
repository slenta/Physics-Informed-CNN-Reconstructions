#!/bin/bash

#SBATCH -J crai-eval
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=480G
#SBATCH --constraint a100_80
#SBATCH --output=/work/uo1075/u241308/ML_downscaling/code/climatereconstructionAI/input/slurm-scripts/eval_slurm_output/eval-cnn.o%j 

module load python3/2022.01-gcc-11.2.0
cd /work/uo1075/u301617/Master_Arbeit/code-goratz/

source activate crai
python -m climatereconstructionai.evaluate --load-from-file ./input/levante/eval-cnn.txt 
