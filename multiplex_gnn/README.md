# Recurrence multiplex graph network (complementary analysis, Appendix B)

An independent graph-network analysis run on GPU (Baobab HPC, University of Geneva), used in the manuscript as
a complementary check on the main graph-neural-network results.

**Question.** Does sensor-level recurrence information improve the prediction of the affective condition of
unseen videos beyond correlation and synchronization?

**Model.** A three-layer multiplex on the 30 sensors: absolute 4–45 Hz Spearman correlation, alpha-band wPLI,
and a sensor recurrence-similarity layer (Jaccard overlap of within-sensor recurrence patterns). Optional
node features: recurrence rate, determinism, laminarity, mean and maximum diagonal length, and line entropy.
Two graph-convolution steps (16 hidden units) with sensor-aligned inter-layer mixing, mean pooling and a
linear classifier; 40 fixed epochs; seeds 11, 29 and 47. Four ablations: connectivity, recurrence + RQA,
connectivity + RQA, and the full multiplex + RQA. A physics-informed variant replaces the learned adjacency
with a scalp-distance graph (`recurrence_multiplex/train_informed.py`).

**Evaluation.** Three video folds (each trial predicted once); 99 whole-video permutations of the
condition labels preserving class counts, refitting every model and seed. Primary contrast: full multiplex
minus connectivity balanced accuracy; Holm correction over five tests.

## Layout

```
core/analysis_core.py                 signal loading and connectivity utilities
multiplex_prediction/                 multiplex model and training loop (experiment.py, model.py)
recurrence_multiplex/                 recurrence features, training, controls, summaries, Slurm scripts
  extract.py / extract.slurm          per-trial recurrence features (48 array tasks)
  train.py / train.slurm              observed labels and permutations (100 array tasks)
  train_control.py, control.slurm     randomized-graph control
  train_informed.py, informed.slurm   physics-informed (scalp-distance) adjacency
  metric_increment.py                 increment of recurrence metrics over synchronization
  summarize.py, *_summary.py          summaries and permutation tests
  cluster/setup.slurm                 environment setup
```

## Running

Set `EEG_PROJECT` (workspace) and `EEG_PYTHON` (Python executable). Place the FACED full-trial export at
`EEG-data-raw-multiscale/full_trial` and the input quality-control table at `inputs/disjoint_signed/input_qc.csv`.
Submit `recurrence_multiplex/extract.slurm`, then `recurrence_multiplex/train.slurm` with an `afterok`
dependency, then:

```bash
python recurrence_multiplex/summarize.py recurrence_multiplex/pilot99
```

Requirements: numpy, pandas, scipy, scikit-learn, statsmodels, torch, joblib, networkx, gudhi.
