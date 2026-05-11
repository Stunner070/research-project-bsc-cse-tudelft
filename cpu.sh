#!/bin/bash
#SBATCH --job-name=testrun
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=01:10:00

srun source ~/projects/.venv/bin/activate

srun python ~/projects/rpbsc/src/scripts/run_full_pipeline.py --mode all