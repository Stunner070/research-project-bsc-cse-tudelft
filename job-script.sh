#!/bin/bash
#SBATCH --job-name=testrun
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=03:00:00

srun source ~/projects/.venv/bin/activate

srun cd ~/projects/rpbsc
srun python run_full_pipeline.py --mode event_frames_only  --device cuda --max_samples 1