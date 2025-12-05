#!/bin/bash

#SBATCH -J spei_calc 
#SBATCH -p compute 
#SBATCH --account=bm1159
#SBATCH --time=08:00:00
#SBATCH --mem=480G
#SBATCH --output=outputs/spei_job.o%j 

module load python3/2023.01-gcc-11.2.0
cd /work/bk1318/k202208/crai/hindcast-pp/Physics-Informed-CNN-Reconstructions/spei-calculation/
source activate crai

lead_year=$1

python cwb_calc.py --lead-year $lead_year
