#!/bin/bash
#SBATCH --job-name=eeg-informed
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-99%4
set -euo pipefail
: "${EEG_PROJECT:?Set EEG_PROJECT to the project directory}"
: "${EEG_PYTHON:?Set EEG_PYTHON to an environment Python executable}"
: "${EEG_MODULE_SETUP:?Set the verified module-loading script path}"
source "$EEG_MODULE_SETUP"
cd "$EEG_PROJECT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
"$EEG_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in allocated GPU job"; x=torch.randn(32,32,device="cuda",requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print("GPU computation check passed:",torch.cuda.get_device_name(),flush=True)'
"$EEG_PYTHON" recurrence_multiplex/train_informed.py --features recurrence_multiplex/features --output recurrence_multiplex/informed99 --permutation "$SLURM_ARRAY_TASK_ID"
