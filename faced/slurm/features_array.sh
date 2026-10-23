#!/bin/bash
#SBATCH --job-name=eeg30s_feat
#SBATCH --partition=shared-cpu
#SBATCH --array=0-47
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/feat_%A_%a.out

# 48 chunks x ~72 trials; each trial ~25-40 CPU-seconds.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python src/01_trial_features.py --chunk "$SLURM_ARRAY_TASK_ID" 48
