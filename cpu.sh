#!/bin/bash
#SBATCH --job-name=testrun2
#SBATCH --partition=compute
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=03:00:00

module load 2025
module load python/3.11.9
module load py-scikit-learn

. /scratch/sofyanali/.venv/bin/activate

#srun python /scratch/sofyanali/rpbsc/src/scripts/run_full_pipeline.py --mode event_frames_only --backbone facenet --privacy_eval
srun /scratch/sofyanali/.venv/bin/python /scratch/sofyanali/rpbsc/run_retrieval_pipeline.py