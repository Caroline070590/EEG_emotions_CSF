#!/bin/bash
#SBATCH --job-name=eeg-env
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
: "${EEG_PROJECT:?Set the absolute remote project path}"
: "${EEG_MODULE_SETUP:?Set the absolute path to the verified module-loading script}"
source "$EEG_MODULE_SETUP"
EEG_ENV="$EEG_PROJECT/.venv"
python -m venv "$EEG_ENV"
"$EEG_ENV/bin/python" -m pip install -r "$EEG_PROJECT/recurrence_multiplex/cluster/requirements.txt"
"$EEG_ENV/bin/python" -m pip check
"$EEG_ENV/bin/python" -m pip freeze > "$EEG_PROJECT/environment.freeze.txt"
cd "$EEG_PROJECT/recurrence_multiplex"
OPENBLAS_NUM_THREADS=1 "$EEG_ENV/bin/python" -m unittest test_recurrence -v
