#!/bin/bash 
#SBATCH --job-name=gp      # create a short name for your job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # number of processes
#SBATCH --mem-per-cpu=4GB (use share ram)
#SBATCH --time=1:00:00 
#SBATCH --output=output.log
#SBATCH --error=error.log

module use /storage/icds/RISE/sw8/modules
module load anaconda/2021.11
conda activate ebf11

python parallel.py 