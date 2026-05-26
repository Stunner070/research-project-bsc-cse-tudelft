#!/bin/bash
#SBATCH --job-name=cutoff_100_300clips
#SBATCH --partition=gpu
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=05:00:00

module load 2025
module load python/3.11.9
module load py-scikit-learn

rm /scratch/sofyanali/.venv/bin/python*
ln -s `which python` /scratch/sofyanali/.venv/bin/python
ln -s `which python3` /scratch/sofyanali/.venv/bin/python3


. /scratch/sofyanali/.venv/bin/activate


#srun python /scratch/sofyanali/rpbsc/src/scripts/run_full_pipeline.py --mode event_frames_only  --device cuda --backbone facenet --privacy_eval
srun python /scratch/sofyanali/rpbsc/run_retrieval_pipeline.py