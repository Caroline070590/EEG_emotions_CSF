#!/bin/bash
#SBATCH --job-name=eeg-null-report
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
source "$EEG_MODULE_SETUP"
cd "$EEG_PROJECT"
export OPENBLAS_NUM_THREADS=1
"$EEG_PYTHON" recurrence_multiplex/summarize.py recurrence_multiplex/pilot99
if [[ "${EEG_REPORT_CONTROL:-0}" == 1 ]]; then
  "$EEG_PYTHON" recurrence_multiplex/control_summary.py
fi
