#!/bin/bash

#SBATCH -J part_38
#SBATCH -p gpu
#SBATCH --reservation=phase2-gpu 
#SBATCH --exclusive
#SBATCH --account=uo1075
#SBATCH --time=10:00:00
#SBATCH --mem=400G
#SBATCH --output=my_train_job.o%j 

module load python3/2022.01-gcc-11.2.0
cd /work/uo1075/u301617/Master_Arbeit/code-goratz/climatereconstructionAI/

source /work/uo1075/u301617/py_envs/py_torch_env/bin/activate
 python train.py \
 --save_part 'part_44' --mask_year '1958_2021_newgrid' --im_name 'Image_' --im_year 'r8_12_newgrid' --image_size 128 --max_iter 1300000 --in_channels 3 --out_channels 3 --device 'cuda' --prepro_mode 'none' --val_interval 5000 --vis_interval 25000 --save_model_interval 25000 --encoding_layers 7 --pooling_layers 0 --attribute_depth 'no_depth' --log_interval 50 --n_threads 16 --attribute_argo 'argo' --mask_argo 'full' --ensemble_member 2 --n_filters 64 --depth 4 --resume_iter 1275000
~                                                                          
