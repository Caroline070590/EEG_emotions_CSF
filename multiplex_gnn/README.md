# Recurrence multiplex graph network (complementary analysis, Appendix B)

An independent graph-network analysis, used in the manuscript as a complementary check on the main
graph-neural-network results. Training uses a GPU.

**Question.** Does sensor-level recurrence information improve the prediction of the affective condition of
unseen videos beyond correlation and synchronization?

**Model.** A three-layer multiplex on the 30 sensors: absolute 4–45 Hz Spearman correlation, alpha-band wPLI,
and a sensor recurrence-similarity layer (Jaccard overlap of within-sensor recurrence patterns). Optional
node features: recurrence rate, determinism, laminarity, mean and maximum diagonal length, and line entropy.
Two graph-convolution steps (16 hidden units) with sensor-aligned inter-layer mixing, mean pooling and a
linear classifier; 40 fixed epochs; seeds 11, 29 and 47. Four ablations: connectivity, recurrence + RQA,
connectivity + RQA, and the full multiplex + RQA. A physics-informed variant replaces the learned adjacency
with a scalp-distance graph.

**Evaluation.** Three video folds (each trial predicted once); 99 whole-video permutations of the condition
labels preserving class counts, refitting every model and seed. Primary contrast: full multiplex minus
connectivity balanced accuracy; Holm correction over five tests.

## Layout

| Path | Content |
|---|---|
| `core/analysis_core.py` | signal loading and connectivity utilities |
| `multiplex_prediction/` | multiplex model and training loop |
| `recurrence_multiplex/extract.py` | per-trial recurrence features |
| `recurrence_multiplex/train.py` | training with observed labels (`--permutation 0`) and permutations |
| `recurrence_multiplex/train_control.py` | randomized-graph control |
| `recurrence_multiplex/train_informed.py` | physics-informed (scalp-distance) adjacency |
| `recurrence_multiplex/metric_increment.py` | increment of the recurrence metrics over synchronization |
| `recurrence_multiplex/summarize.py`, `*_summary.py` | summaries and permutation tests |
| `recurrence_multiplex/test_recurrence.py` | unit tests of the recurrence measures |
| `results/pilot99/` | scores of the observed run (`scores_0000.json`) and the 99 permutations, and `SUMMARY.json` |
| `results/graph_control99/` | randomized-graph control, observed and 99 permutations, and `RESULTS.json` |

## Running

From `multiplex_gnn/`, with the FACED full-trial export at `EEG-data-raw-multiscale/full_trial` and the input
quality-control table at `inputs/disjoint_signed/input_qc.csv`:

```bash
# 1. recurrence features (the 3,444 trials can be split into shards: --shard i --shards n)
python recurrence_multiplex/extract.py --shard 0 --shards 1 --data-root . --output recurrence_multiplex/features

# 2. observed labels (0) and 99 permutations (1-99)
for k in $(seq 0 99); do
  python recurrence_multiplex/train.py --features recurrence_multiplex/features --output recurrence_multiplex/pilot99 --permutation $k
done

# 3. summary and permutation tests
python recurrence_multiplex/summarize.py recurrence_multiplex/pilot99
```

The physics-informed variant is reported in the manuscript in summary form only. The control and physics-informed variants are run in the same way with `train_control.py` and
`train_informed.py` (outputs `graph_control99` and `informed99`), followed by `control_summary.py` and
`informed_summary.py`.

Requirements: numpy, pandas, scipy, scikit-learn, statsmodels, torch, joblib, networkx, gudhi.
