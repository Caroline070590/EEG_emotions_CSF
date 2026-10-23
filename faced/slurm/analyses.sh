#!/bin/bash
#SBATCH --job-name=faced_stats
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/analyses_%j.out

# Run from faced/ after features_array.sh and rqa_array.sh have finished:
# merges the feature chunks, then computes all statistics, sensitivity analyses and figures.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ./env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python src/01_trial_features.py --merge 48
python src/02_recurrence_features.py --merge 48
cd src
for s in 03_stimulus_aware_contrasts.py \
         fig1_fig2_synchronization_and_stimulus.py fig3_decoding_unseen_videos.py \
         figS1_quantile_density_sweep.py figS2_coupling_null_models.py figS3_emg_checks.py \
         figS4_signed_nbs.py figS5_persistent_homology.py figS6_recurrence_statistics.py \
         sensitivity_qg_lag.py sensitivity_normalization.py; do
  echo "===== $s"; python "$s"
done
