#!/bin/bash
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=eegnet_gpu_env
set -euo pipefail
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$HOME/scratch/eeg_matched_2026-10-24/pip_cache"
EEG_GPU_ENV="$HOME/scratch/conda_envs/eegnet_cuda126_20261024"
if [ ! -x "$EEG_GPU_ENV/bin/python" ]; then
  "$HOME/scratch/conda_envs/eeg_revision/bin/python" -m venv --system-site-packages "$EEG_GPU_ENV"
fi
"$EEG_GPU_ENV/bin/python" -m pip install --upgrade 'torch==2.8.0+cu126' --index-url https://download.pytorch.org/whl/cu126
"$EEG_GPU_ENV/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda == "12.6"'
