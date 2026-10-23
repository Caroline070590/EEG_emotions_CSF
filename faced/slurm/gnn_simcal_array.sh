#!/bin/bash
#SBATCH --job-name=gnn_simcal
#SBATCH --partition=shared-cpu
#SBATCH --array=1-126
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/simcal_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source ./env.sh
export OMP_NUM_THREADS=2
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/gnn_simcal_params.txt)
python src/gnn/run_gnn.py $ARGS --threads 2 --out results/gnn/simcal_results.jsonl
