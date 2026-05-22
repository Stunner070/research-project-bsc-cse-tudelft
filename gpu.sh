#!/bin/bash
#SBATCH --job-name=testrun
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=01:10:00

. /scratch/sofyanali/.venv/bin/activate

srun python /scratch/sofyanali/rpbsc/src/scripts/run_full_pipeline.py --mode event_frames_only  --device cuda --backbone facenet --privacy_eval