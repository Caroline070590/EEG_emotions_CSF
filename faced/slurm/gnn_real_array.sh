#!/bin/bash
#SBATCH --job-name=gnn_real
#SBATCH --partition=shared-cpu
#SBATCH --array=1-608%150
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gnnreal_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source ./env.sh
export OMP_NUM_THREADS=2
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/gnn_real_params.txt)
case "$ARGS" in *simcal*) OUT=results/gnn/simcal_results.jsonl ;; *) OUT=results/gnn/real_results.jsonl ;; esac
python src/gnn/run_gnn.py $ARGS --threads 2 --out $OUT
