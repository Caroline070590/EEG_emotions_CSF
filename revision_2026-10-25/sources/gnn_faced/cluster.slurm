#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --job-name=eeg_participant

set -euo pipefail
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/eeg_participant_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd "$SLURM_SUBMIT_DIR"
"$HOME/scratch/conda_envs/eeg_revision/bin/python" cluster.py "$1" \
  --cache "$HOME/EEG_emotion_revision_2026-10/cache" \
  --out-dir "$SLURM_SUBMIT_DIR/cluster_results"
