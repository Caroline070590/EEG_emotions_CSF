# Reproducibility additions, 25 October 2026

This accompanying archive contains the frozen analyses and consistency corrections used in the revised manuscript. Raw EEG and prepared signal arrays are excluded. The retained graph matrices, scalar features, behavioral outcomes, and predictions are derived analysis inputs/outputs.

## Scientific map

| Item | Source | Retained evidence |
|---|---|---|
| Main Fig. 2 synchronization and coupling | `corrections/analyses/figure4_s15/render_synchrony.py` | Verified 3,444-trial coupling coefficients, original plot/statistics source, two-page readable rendering |
| Main Fig. 3 multiscale/regional networks | `corrections/analyses/figures/part4_regional/run_figure5_corrected.py` | Derived graph inputs, regional coefficient table, readable two-page rendering; historical filename contains Figure5 |
| S15 NBS figure | `corrections/analyses/figure4_s15/render_nbs.py` | Preserved original PDF, derived network inputs, edge counts and component statistics |
| One-second wPLI estimator | `corrections/analyses/wpli/` | All 3,444 raw-input checks; 240 recovered coupling coefficients; band/regional/behavioral lineage; regression tests |
| Behavioral J4 prediction and S28 panel | `corrections/analyses/behavioral/correct_prediction.py` | Per-fold training-only outcome normalization, all predictions/scalers/errors, both outcome scales, paired tests |
| SEED source-held-out session sensitivity | `corrections/analyses/seed_session/` | All 1,000 session-composition-preserving source-label assignments, refit scores, 54 source folds, derived feature snapshot |
| Table S2, participant-held-out GNN/LDA | `sources/gnn_faced/`, `sources/gnn_seed7/` | All observed runs, 99 full-refit null assignments per model, input-matched comparisons, feature-group reliance |
| Table S3, matched EEGNet and all seeds | `sources/matched_eegnet/` | All 300 fold fits, five models, two datasets, two window designs, three seeds; 16 primary paired comparisons |
| Table S4, preprocessing sensitivity | `sources/eegnet_preprocessing_diagnostic/` | All nine post-result diagnostic fits and their descriptive comparisons |

The source archives, original deployment manifests, and frozen configurations are retained byte-for-byte under `results/frozen_sources/`. The GNN archives were reconstructed from individually hash-matching deployment files; they are not represented as original archive bytes. The readable `sources/` copies do not alter these frozen experiments.

## Reproduction

Run from this directory with the environments recorded by the corresponding source bundle. To regenerate prediction reports from the bundled OOF predictions (this creates new `report/` directories under the retained runs):

```bash
python sources/gnn_faced/report.py --out-dir retained_runs/gnn_faced
python sources/gnn_seed7/report.py --out-dir retained_runs/gnn_seed7
python sources/matched_eegnet/report.py --out-dir retained_runs/matched_eegnet
```

The immutable supplied reports remain in `results/`. Some run records also reference training checkpoints/scalers and split NPZs held in the original execution environment; these large training artifacts are not included. OOF predictions, fold subject lists, selected epochs, config/source hashes, and loss curves are included. Re-training requires separately obtained EEG and new output directories, with input paths adapted without presenting the altered configuration as the original frozen run.

Focused corrected-estimator regression checks:

```bash
python -m pytest corrections/analyses/wpli/test_corrected_synchrony.py corrections/analyses/wpli/test_corrected_coupling.py
python corrections/analyses/behavioral/verify_saved_outputs.py
python -m pytest corrections/analyses/behavioral/test_prediction.py
```

The behavioral analysis is self-contained from the bundled validated derived inputs. Use its `--out` option to create a fresh result directory; existing results are protected. `faced_1s_exploratory/src/behavioral/behavioral_grounding.py` in the older release section is historical source for the complete behavioral package; its old prediction normalization is superseded by this corrected J4 implementation. The unchanged association and RSA results are not newly certified by the J4 correction.

For SEED, `corrections/analyses/seed_session/refit_snapshot.py` independently reconstructs feature normalization and reproduces the observed fit and three selected saved null assignments. With `--all` it refits all 1,000 assignments. This uses the bundled derived feature snapshot and no raw EEG. The separate historical `verify_session_control.py` also checks the original absolute cache paths and therefore requires the original input layout; its retained verification record documents that original-input check. The source-fold session sensitivity conditions on session composition and leaves seven sources fixed; it is neither an emotion-specificity proof nor a replacement for the participant-held-out GNN/EEGNet null.

The three corrected figure renderers use their bundled derived inputs. Historical raw-lineage audit scripts preserve their original workspace paths and source hashes. Their numerical outputs and portable focused regression tests are supplied; rerunning all raw-lineage checks requires the original data export and corresponding source/cache layout described in `corrections/analyses/wpli/PROVENANCE_AND_SCOPE.md`.

## Interpretation

Balanced-accuracy improvements relative to a model's label null are different from relative gains over nominal uniform guessing. Participant-held-out models concern familiar videos and sessions in new participants. Source-held-out models address a different generalization target. The matched EEGNet primary comparison uses all 30 seconds in both designs: either three nonoverlapping 10-s probabilities averaged per trial or one 30-s input. Additional engineered features describe the same EEG and do not establish independent physiology or validate transition-network topology.

Participant bootstraps and paired sign flips condition on fitted predictions. They do not reproduce uncertainty from re-training, and the overlapping outer-training samples limit the paired sign-flip approximation. Seed comparisons and preprocessing diagnostics retain every planned run, rather than selecting the best result. See `REVIEW_CHECKLIST.md` for explicit provenance and validation limits.

## Subsequent critical review

The Figure 1 q=30 entrypoint now includes the recovered historical plotting helpers and validates q=30 when accepting current cache filenames. A portable test and derived fixtures are in `provenance/critical_review/figure1/`. All 162 original fingerprint cells and the plotting helper’s participant bootstrap intervals were verified; the manuscript figure was not numerically replaced. The audit also refitted all 144 scalar topology contrasts, verified actual coupling sample counts and all displayed simplex/NBS component counts, and clarified descriptive versus inferential scope.
