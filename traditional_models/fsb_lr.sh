#!/bin/sh
#SBATCH --partition=graceALL
#SBATCH --exclude=ethnode[22]
#SBATCH --job-name=fsb_lr
#SBATCH --output=../../../scratch/s2034697/slurm_output/%x_%j.out

START=$SECONDS

echo "[$SHELL] #### Starting script"
python3 -m  logreg_model
echo "[$SHELL] ## Script finished"

FINISH=$SECONDS

timer () 
{
hrs="$((($FINISH - $START)/3600))hrs"
min="$(((($FINISH - $START)/60)%60))min"
sec="$((($FINISH - $START)%60))sec"

if [[ $(($FINISH - $START)) -gt 3600 ]]; then echo "$hrs, $min, $sec"
elif [[ $(($FINISH - $START)) -gt 60 ]]; then echo "$min, $sec"
else echo "$sec"
fi
}

echo -n "Elapsed time: "; timer
