#!/bin/bash
#SBATCH --job-name=eeg-recurrence
#SBATCH --partition=shared-cpu
#SBATCH --array=0-47%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail
: "${EEG_PROJECT:?Set EEG_PROJECT to the project directory}"
: "${EEG_PYTHON:?Set EEG_PYTHON to an environment Python executable}"
: "${EEG_MODULE_SETUP:?Set the verified module-loading script path}"
source "$EEG_MODULE_SETUP"
cd "$EEG_PROJECT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$EEG_PYTHON" recurrence_multiplex/extract.py --shard "$SLURM_ARRAY_TASK_ID" --shards 48 --output recurrence_multiplex/features --data-root "$EEG_PROJECT"
