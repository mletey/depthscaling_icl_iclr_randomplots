#!/bin/bash
# MHA_T20000_lr0p0001_isotropic.sbatch
# 
#SBATCH --job-name=MHA_T20000_lr0p0001_isotropic
#SBATCH -n 1                
#SBATCH -N 1               
#SBATCH -t 0-2:00:00   
#SBATCH --array=1-5
#SBATCH -p kempner  
#SBATCH --account kempner_pehlevan_lab
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH -o log_files/MHA_T20000_lr0p0001_isotropic_%a.out 
#SBATCH -e log_files/MHA_T20000_lr0p0001_isotropic_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module purge
module load python/3.10.12-fasrc01
source activate try4

parentdir="rebuttals"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"

python SOFTMAX_UNROLLED.py $newdir $SLURM_ARRAY_TASK_ID 20000 0.0001 2 0.0