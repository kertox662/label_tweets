#!/bin/bash
#SBATCH --account=def-jhoey
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --time=0-24:00
#SBATCH --mem=80G
#SBATCH --job-name=self-supervised-08-20
#SBATCH --output=logs/%x-%A-%a.out  # Optional: output log per array job/task

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load python StdEnv/2023 intel/2023.2.1 cuda/11.8 scipy-stack/2023b
module load gcc arrow/19.0.1
source ~/projects/def-jhoey/atjhin/label_tweets/venv/bin/activate

python -m Refactored.self_supervised_learning ${SLURM_ARRAY_TASK_ID}
