#!/bin/bash
#SBATCH --job-name=eeg_rqa
#SBATCH --partition=shared-cpu
#SBATCH --array=0-47
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rqa_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source ./env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python src/02_recurrence_features.py --chunk "$SLURM_ARRAY_TASK_ID" 48
