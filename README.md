# EEG network organization across affective video conditions

Code accompanying

> Alves C. L., Kalbasi A., Saccaro L. F., Andrianarivelo N., Panzeri A., Bellone C.
> *EEG network organization across affective video conditions: complementary nonlinear representations,
> topological summaries, and synchronization–topology coupling.* Chaos, Solitons & Fractals (under review).

The study asks which EEG network and dynamical descriptors of naturalistic affective stimulation
generalize beyond the specific stimuli. It analyses band-specific synchronization, the topology of three
complementary network representations (Spearman channel networks, Markov-transition-field and
quantile-graph state-transition networks), persistence-based summaries, recurrence quantification and
graph neural networks, with participants **and** stimuli treated as random factors, and replicates the
main analyses in an independent dataset (SEED-VII).

## Repository structure

| Folder | Content | Manuscript |
|---|---|---|
| [`faced/`](faced) | FACED, released 30-s epochs: features, stimulus-aware statistics, null models, recurrence quantification, decoding of unseen videos, graph neural networks | Sections 2, 3.1; Appendices A–B; Figs. 1–3, S1–S7 |
| [`seed7/`](seed7) | Independent replication in SEED-VII with the same code; analysis plan and deviation log | Section 3.2; Appendix C; Fig. S8 |
| [`faced_1s_exploratory/`](faced_1s_exploratory) | Exploratory analyses of the final 1-s segment of each FACED trial (the analyses of the original submission) and the behavioral analyses | Sections 3.3–3.5; Appendices D–K; Figs. 4–6, S9–S29 |
| [`multiplex_gnn/`](multiplex_gnn) | Complementary recurrence-multiplex graph network (GPU) | Appendix B |

Each analysis folder contains `src/` (code), `figures/` and `tables/` (the outputs reported in the paper).

## Data

No EEG data are included.

* **FACED** (Chen et al., 2023, *Scientific Data* 10, 740; doi:10.1038/s41597-023-02650-w): 123 participants,
  28 videos, 9 conditions, 30 channels at 250 Hz. The 30-s analyses read the released preprocessed epochs
  exported as one CSV per participant × video (`faced/src/00_load_epochs.py`, environment variable
  `FACED_ZIP`); the 1-s analyses read the final second of each epoch (`<emotion>/Patient###/trial_*.csv`,
  see `faced_1s_exploratory/src/validate_inputs.py`).
* **SEED-VII** (Jiang et al., 2025, *IEEE Trans. Affect. Comput.* 16, 969; doi:10.1109/TAFFC.2024.3485057):
  available on request from the BCMI laboratory (https://bcmi.sjtu.edu.cn/~seed/seed-vii.html) under a license
  that does not permit redistribution. The replication uses the official preprocessed EEG.

Intermediate caches (`cache/`) are created by the scripts and are not version-controlled.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # FACED 30-s and SEED-VII analyses
pip install -r faced_1s_exploratory/requirements.txt  # 1-s exploratory analyses
```

## Reproducing the analyses

**FACED, 30-s epochs** (run from `faced/`). On a Slurm cluster, set `EEG_ENV` to the Python environment and
submit `slurm/features_array.sh` and `slurm/rqa_array.sh`, then `slurm/analyses.sh`. Equivalent steps on one
machine:

```bash
cd faced/src
python 00_load_epochs.py                 # 3,444 epochs, 30 channels x 7,500 samples
python 01_trial_features.py              # synchronization, network and topology features
python 02_recurrence_features.py --chunk 0 1 && python 02_recurrence_features.py --merge 1
python 03_stimulus_aware_contrasts.py    # 224-test primary family and exploratory families
python fig1_fig2_synchronization_and_stimulus.py
python fig3_decoding_unseen_videos.py
python figS1_quantile_density_sweep.py   # ... and the other figS*.py and sensitivity_*.py scripts
```

Graph neural networks: `src/gnn/run_gnn.py` (one run per call; the parameter grids are in `slurm/*_params.txt`),
`src/gnn/summarize_paired.py` and `src/gnn/figS7_gnn_models.py`. All runs, including every permutation, are in
`results/gnn/`. The model-selection rule and its amendments, each recorded before the corresponding
real-data run, are in `preregistration/gnn_model_selection_rule.md`.

**SEED-VII** (run from `seed7/`): `./run_all.sh /path/to/SEED-VII`. The plan written before the data were
analyzed, with a log of every later addition, is in `seed7/ANALYSIS_PLAN.md`.

**FACED, final 1-s segments** (run from `faced_1s_exploratory/src/`, with the 1-s export at `./EEG-data-raw`):
`python validate_inputs.py --data-root EEG-data-raw`, then `python run_all.py --n_jobs -1 --n_rand 20 --n_boot 2000`
for the base pipeline and the individual scripts listed below for the remaining figures.

## Figures

| Figure | Script |
|---|---|
| 1, 2 | `faced/src/fig1_fig2_synchronization_and_stimulus.py` |
| 3 | `faced/src/fig3_decoding_unseen_videos.py` |
| 4 | `faced_1s_exploratory/src/part1_topology/run_part1_q30.py` |
| 5 | `faced_1s_exploratory/src/part2_synchrony/run_part2.py` |
| 6 | `faced_1s_exploratory/src/part4_regional/run_figure4_full_composite.py` (panels from `part6_advanced/run_nbs.py`, `run_centrality_topo.py`, `run_gudhi_simplices.py`, `part4_regional/run_fig4_panel_f_combined.py`) |
| S1 | `faced/src/figS1_quantile_density_sweep.py` |
| S2 | `faced/src/figS2_coupling_null_models.py` |
| S3 | `faced/src/figS3_emg_checks.py` |
| S4 | `faced/src/figS4_signed_nbs.py` |
| S5 | `faced/src/figS5_persistent_homology.py` |
| S6 | `faced/src/figS6_recurrence_statistics.py` |
| S7 | `faced/src/gnn/figS7_gnn_models.py` |
| S8 | `seed7/src/figS8_compare_with_faced.py` |
| S9 | `faced_1s_exploratory/src/part6_advanced/run_density_sweep_q30.py` |
| S10 | `faced_1s_exploratory/src/part1_topology/run_chords_supp_q30.py` |
| S11 | `faced_1s_exploratory/src/part1_topology/run_bootstrap_effects.py` |
| S12 | `faced_1s_exploratory/src/part1_topology/run_pairwise.py` |
| S13 | `faced_1s_exploratory/src/part3_embedding_topology/run_umap_q30.py` |
| S14 | `faced_1s_exploratory/src/part6_advanced/run_splithalf_q30_consolidated.py` |
| S15 | `faced_1s_exploratory/src/part6_advanced/run_nbs.py` |
| S16, S17 | `faced_1s_exploratory/src/part6_advanced/run_centrality_topo.py` |
| S18 | `faced_1s_exploratory/src/part6_advanced/run_nbs_threshold_sensitivity.py` |
| S19 | `faced_1s_exploratory/src/part6_advanced/run_region_coupling_splithalf.py` |
| S20 | `faced_1s_exploratory/src/part4_regional/run_part4_mtf_only_fancy_v2.py` |
| S21 | `faced_1s_exploratory/src/part5_enrichment/run_network_enrichment.py` |
| S22 | `faced_1s_exploratory/src/part6_advanced/run_richclub.py`, `run_richclub_mtf_q30.py` |
| S23 | `faced_1s_exploratory/src/part6_advanced/run_participant_averaging_sensitivity.py` |
| S24 | generator not recovered (leave-one-participant-out omnibus stability) |
| S25 | `faced_1s_exploratory/src/part6_advanced/run_contrast_splithalf.py` |
| S26 | `faced_1s_exploratory/src/stimulus_aware/stimulus_aware_validation_pipeline.py` |
| S27–S29 | `faced_1s_exploratory/src/behavioral/behavioral_grounding.py` |

The figures generated by the 30-s, SEED-VII and regenerated 1-s scripts are included in the corresponding
`figures/` folders; the other 1-s figures are produced by the scripts above.

## Tables

`faced/tables/all_contrasts.csv` lists every FACED emotion-versus-neutral test with uncorrected and
false-discovery-rate-adjusted values under both inference schemes (participants random; participants and
videos random). `seed7/tables/all_contrasts.csv` is the SEED-VII counterpart. The remaining files in each
`tables/` folder hold the results of the individual analyses and are named after them.

## Notes on the statistics

* Participants and stimuli are treated as crossed random factors with the conservative min F' approach
  (Clark, 1973) in `faced/src/lib_rev.py` and `seed7/src/lib_rev.py`.
* Decoding of unseen stimuli is evaluated against stimulus-structured permutation nulls; every model used
  for inference is compared with its own null.
* Within-participant feature standardization uses all of a participant's stimuli (no labels). Refitting it on
  training stimuli only is reported in `faced/src/sensitivity_normalization.py` and
  `seed7/src/07_normalization_sensitivity.py`.
