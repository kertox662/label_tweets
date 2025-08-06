#!/bin/bash
#SBATCH --account=def-jhoey
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00
#SBATCH --job-name=test_2
#SBATCH --output=logs/%x-%A-%a.out  # Optional: output log per array job/task

module load python
source ~/projects/def-jhoey/atjhin/label_tweets/venv/bin/activate

python -m Refactored.main ${SLURM_ARRAY_TASK_ID}
