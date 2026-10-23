#!/bin/bash
# SEED-VII replication, full run on one machine. usage (from seed7/): ./run_all.sh /path/to/SEED-VII
# SEED-VII must contain EEG_preprocessed/<subject>.mat and the metadata spreadsheets of the official release.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
cd "$(dirname "$0")/src"
python 00_load_clips.py "$1"                    # 1,600 trials: last 30 s, FACED channels, 250 Hz
python 01_trial_features.py                     # synchronization and topology (primary family)
python 02_recurrence_features.py --all          # recurrence quantification
python 03_stimulus_aware_contrasts.py           # emotion vs neutral, clips fixed / random
python 04_source_nested_contrasts.py            # source programmes as the stimulus unit (nested)
python 05_decoding_unseen_sources.py 1000 source  # leave-one-source-out decoding, 1,000 permutations
python 05_decoding_unseen_sources.py 200 clip     # leave-one-clip-out decoding (secondary), 200 permutations
python 06_decoding_source_block_null.py 1000    # stricter source-by-condition block null
python 07_normalization_sensitivity.py          # standardization fitted on training clips only
python 08_recurrence_delta_adjustment.py        # exploratory: recurrence adjusted for relative delta power
python figS8_compare_with_faced.py              # comparison with FACED (Supplementary Fig. S8)
