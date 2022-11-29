#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --mem=1GB 
#SBATCH --time=1:00:00 
#SBATCH --partition=open 

module load python/3.8
python Song.py