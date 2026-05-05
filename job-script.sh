#!/bin/bash
#SBATCH --job-name=testrun
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=00:01:00

srun source ~/projects/.venv/bin/activate

srun cd ~/projects/rpbsc
srun python init.py