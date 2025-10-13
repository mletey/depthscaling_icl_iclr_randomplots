#!/bin/bash
# SOFTMAX_T5000test_lr0p005_isotropic.sbatch
# 
#SBATCH --job-name=SOFTMAX_T5000test_lr0p005_isotropic
#SBATCH -n 1                
#SBATCH -N 1               
#SBATCH -t 0-16:00   
#SBATCH --array=1-5
#SBATCH -p kempner  
#SBATCH --account kempner_pehlevan_lab
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH -o log_files/SOFTMAX_T5000test_lr0p005_isotropic_%a.out 
#SBATCH -e log_files/SOFTMAX_T5000test_lr0p005_isotropic_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module purge
module load python/3.10.12-fasrc01
source activate try4

parentdir="outputs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"

python SOFTMAX_UNROLLED.py $newdir $SLURM_ARRAY_TASK_ID 5000 0.005