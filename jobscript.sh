#!/bin/bash
#SBATCH --account=def-jhoey
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-8:00
#SBATCH --mem=32G
#SBATCH --job-name=test_2
#SBATCH --output=logs/%x-%A-%a.out  # Optional: output log per array job/task

module load python StdEnv/2023 intel/2023.2.1 cuda/11.8 scipy-stack/2023b
module load gcc arrow/19.0.1
source ~/projects/def-jhoey/atjhin/label_tweets/venv/bin/activate

python -m Refactored.main ${SLURM_ARRAY_TASK_ID}
