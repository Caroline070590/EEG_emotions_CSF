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
EEG_AUDIT_OUT="$HOME/scratch/eegnet_reference_audit_2026-10-24"
EEG_AUDIT_RAW="$HOME/EEG_emotion_revision_2026-10/cache/eeg_30s.npy"
EEG_AUDIT_CSV="$HOME/EEG_emotion_revision_2026-10/cache/features_scalar.csv"
case "$1" in
  prepare)
    if [[ ! -f "$EEG_AUDIT_OUT/config.json" ]]; then
      "$EEG_CPU_PYTHON" protocol.py --raw "$EEG_AUDIT_RAW" --metadata-csv "$EEG_AUDIT_CSV" --out "$EEG_AUDIT_OUT"
    fi
    "$EEG_CPU_PYTHON" prepare.py --config "$EEG_AUDIT_OUT/config.json" ;;
  smoke)
    "$EEG_GPU_PYTHON" run.py --config "$EEG_AUDIT_OUT/config.json" --task 0 --device cuda --smoke ;;
  train)
    "$EEG_GPU_PYTHON" run.py --config "$EEG_AUDIT_OUT/config.json" --task "${SLURM_ARRAY_TASK_ID:?}" --device cuda ;;
  report)
    "$EEG_CPU_PYTHON" report.py --config "$EEG_AUDIT_OUT/config.json" ;;
  *) echo 'Unknown audit stage' >&2; exit 2 ;;
esac
