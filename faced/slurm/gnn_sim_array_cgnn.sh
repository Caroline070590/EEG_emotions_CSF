#!/bin/bash
#SBATCH --job-name=cgnn_sim
#SBATCH --partition=shared-cpu
#SBATCH --array=1-36
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cgnnsim_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source ./env.sh
export OMP_NUM_THREADS=2
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/gnn_sim_params_cgnn.txt)
python src/gnn/run_gnn.py $ARGS --threads 2 --out results/gnn/sim_results.jsonl
