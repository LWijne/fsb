#!/bin/sh
#SBATCH --partition=graceCPU
#SBATCH --exclude=ethnode[22]
#SBATCH --job-name=fsb_rf
#SBATCH --output=../../../scratch/s2034697/slurm_output/%x_%j.out

echo "[$SHELL] #### Starting script"
python3 -m  ranfor_model
echo "[$SHELL] ## Script finished"
