#!/bin/bash
#SBATCH --job-name=eeg-rqa-increment
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
set -euo pipefail
source "$EEG_MODULE_SETUP"
cd "$EEG_PROJECT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$EEG_PYTHON" recurrence_multiplex/metric_increment.py --features recurrence_multiplex/features --synchrony inputs/incremental_topology/features.npz --output recurrence_multiplex/metric_increment_99
