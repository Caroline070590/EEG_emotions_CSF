#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EEG_CPU_PYTHON="$HOME/scratch/conda_envs/eeg_revision/bin/python"
EEG_GPU_PYTHON="$HOME/scratch/conda_envs/eegnet_cuda126_20261024/bin/python"
EEG_WORK="$HOME/scratch/eeg_matched_2026-10-24"
case "$1" in
  preflight)
    "$EEG_CPU_PYTHON" preflight.py --inputs deployment/input/inputs.json ;;
  feature)
    EEG_TASK=${SLURM_ARRAY_TASK_ID:?}
    EEG_CHUNK=$((EEG_TASK % 64))
    EEG_CONTEXT=$((EEG_TASK / 64))
    EEG_DATASETS=(faced faced seed7 seed7)
    EEG_SECONDS=(10 30 10 30)
    "$EEG_CPU_PYTHON" prepare.py chunk --inputs deployment/input/inputs.json --prepared "$EEG_WORK/prepared" --dataset "${EEG_DATASETS[$EEG_CONTEXT]}" --seconds "${EEG_SECONDS[$EEG_CONTEXT]}" --chunks 64 --chunk "$EEG_CHUNK" ;;
  merge)
    EEG_DATASETS=(faced faced seed7 seed7)
    EEG_SECONDS=(10 30 10 30)
    EEG_TASK=${SLURM_ARRAY_TASK_ID:?}
    "$EEG_CPU_PYTHON" prepare.py merge --inputs deployment/input/inputs.json --prepared "$EEG_WORK/prepared" --dataset "${EEG_DATASETS[$EEG_TASK]}" --seconds "${EEG_SECONDS[$EEG_TASK]}" --chunks 64 ;;
  configure)
    "$EEG_CPU_PYTHON" protocol.py --prepared "$EEG_WORK/prepared" --out results
    "$EEG_CPU_PYTHON" session_control.py --prepared "$EEG_WORK/prepared" --out results ;;
  probe)
    "$EEG_GPU_PYTHON" gpu_probe.py ;;
  smoke)
    "$EEG_GPU_PYTHON" train.py --prepared "$EEG_WORK/prepared" --config results/config.json --out smoke --task 15 --smoke ;;
  train)
    "$EEG_GPU_PYTHON" train.py --prepared "$EEG_WORK/prepared" --config results/config.json --out results --task "${SLURM_ARRAY_TASK_ID:?}" ;;
  report)
    "$EEG_CPU_PYTHON" report.py --out-dir results --bootstrap 2000 --randomizations 9999 ;;
  *) echo 'Unknown pipeline stage' >&2; exit 2 ;;
esac
