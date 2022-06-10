#!/bin/bash

#SBATCH -J SimonLentz
#SBATCH -p gpu
#SBATCH --account=uo1075
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=mg207
#SBATCH --time=12:00:00
#SBATCH --mem=128G

module load cuda/10.0.130
module load singularity/3.6.1-gcc-9.1.0

singularity exec --bind /work/uo1075/u301617/ --nv /work/uo1075/u301617/Master-Arbeit_jm/pytorch_gpu_new.sif \
 python train.py \
 --save_part 'part_2' --mask_year '2001_2020_newgrid' --im_year 'r11_newgrid' --image_size 128 --max_iter 50002 --in_channels 20 --out_channels 20 --device 'cuda' --mode 'none' --val_interval 25000 --vis_interval 25000 --save_model_interval 25000 --encoding_layers 4 --pooling_layers 2 --resume_iter 25000 --depth 
