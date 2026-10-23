#!/bin/bash
#SBATCH --job-name=prior_real
#SBATCH --partition=shared-cpu
#SBATCH --array=1-603%150
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prior_real_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source ./env.sh
export OMP_NUM_THREADS=2
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/prior_real_params.txt)
python src/gnn/run_gnn.py $ARGS --threads 2 --out results/gnn/prior_real_results.jsonl
