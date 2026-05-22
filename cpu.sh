#!/bin/bash
#SBATCH --job-name=testrun
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=03:00:00

. /scratch/sofyanali/.venv/bin/activate

srun python /scratch/sofyanali/rpbsc/src/scripts/run_full_pipeline.py --mode event_frames_only --backbone facenet --privacy_eval