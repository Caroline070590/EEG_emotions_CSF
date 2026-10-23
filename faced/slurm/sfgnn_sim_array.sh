#!/bin/bash
#SBATCH --job-name=sfg_sim
#SBATCH --partition=shared-cpu
#SBATCH --array=1-126%150
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sfg_sim_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source ./env.sh
export OMP_NUM_THREADS=2
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/sfgnn_sim_params.txt)
python src/gnn/run_gnn.py $ARGS --threads 2 --out results/gnn/sfgnn_sim_results.jsonl
