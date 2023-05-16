#!/bin/bash
#SBATCH --job-name=gp_parallel
#SBATCH --output=gp_parallel-%a.out
#SBATCH --error=gp_parallel-%a.errs
#SBATCH --array=1-5  # Specifies the range of tasks to run (from 1 to 5 in this example)
#SBATCH --ntasks=1  # Number of tasks per job
#SBATCH --cpus-per-task=4  # Number of CPU cores per task
#SBATCH --mem-per-cpu=2G  # Memory required per CPU core

# Load any necessary modules or activate the virtual environment
module use /storage/icds/RISE/sw8/modules
module load anaconda/2021.11

# Activate the virtual environment if needed
conda activate ebf11

# Run the Python script with appropriate arguments
python gp_parallel.py $SLURM_ARRAY_TASK_ID