# FACED: analyses of the released 30-s epochs

Sections 2 and 3.1 and Appendices A–B of the manuscript (Figs. 1–3, S1–S7).

## Pipeline (`src/`)

| Step | Script | Output |
|---|---|---|
| 0 | `00_load_epochs.py` | `cache/eeg_30s.npy`: 3,444 epochs × 30 channels × 7,500 samples |
| 1 | `01_trial_features.py` | synchronization (5 bands), Spearman / MTF / QG networks and topology, band power, persistence |
| 2 | `02_recurrence_features.py` | recurrence quantification (DET, LAM, L_mean, ENTR, TT) and joint-recurrence synchronization |
| 3 | `03_stimulus_aware_contrasts.py` | `tables/all_contrasts.csv`, `tables/omnibus_tests.csv` |
| | `fig1_fig2_synchronization_and_stimulus.py` | Figs. 1–2 |
| | `fig3_decoding_unseen_videos.py` | Fig. 3, `tables/decoding_unseen_videos.csv` |
| | `figS1_quantile_density_sweep.py` | Fig. S1 |
| | `figS2_coupling_null_models.py` | Fig. S2 (phase-randomized surrogates, independent halves) |
| | `figS3_emg_checks.py` | Fig. S3 |
| | `figS4_signed_nbs.py` | Fig. S4 |
| | `figS5_persistent_homology.py` | Fig. S5 |
| | `figS6_recurrence_statistics.py` | Fig. S6 |
| | `sensitivity_qg_lag.py`, `sensitivity_normalization.py` | sensitivity analyses |
| | `gnn/run_gnn.py`, `gnn/summarize_paired.py`, `gnn/figS7_gnn_models.py` | graph neural networks, Fig. S7 |

`lib_rev.py` contains the shared statistics: min F′ with participants and videos as crossed random factors
(Clark, 1973), omnibus tests and Benjamini–Hochberg correction.

## Running

Set `FACED_ZIP` to the FACED epochs exported as one CSV per participant × video (channels as columns), then
follow the steps above from `src/`, or on a Slurm cluster submit `slurm/features_array.sh` and
`slurm/rqa_array.sh`, then `slurm/analyses.sh` (set `EEG_ENV` to the Python environment; see `env.sh`).
