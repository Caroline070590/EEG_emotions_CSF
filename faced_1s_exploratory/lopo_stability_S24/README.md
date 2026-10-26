# Supplementary Fig. S24: leave-one-participant-out (LOPO) stability

The original S24 generator was not recovered. Its July 2026 outputs and input tables are kept in `original_2026-07/`
(`FigS_LOPO_headline_effect_stability_cropped.pdf` is the earlier figure). This folder verifies that analysis and recomputes it
on the inputs of the current paper. See `PROTOCOL.md` (written before recomputation) and `RESULTS.md`.

| Step | Script | Needs | Output |
|---|---|---|---|
| 0 | `verify_original.py` | `original_2026-07/` | reproduces all 5,166 original rows: `results/ORIGINAL_REPRODUCTION.json` |
| 1 | `compute_persistence_q30.py` | 1-s export in `../src/EEG-data-raw` | `inputs/tda_{spearman,mtf,qg}_q30.csv` (retained) |
| 2 | `lopo_recompute.py` | step-1 tables, main Fig. 1 topology tables, corrected synchronization table | `results/` |
| 3 | `make_figure.py` | `results/` | `figures/FigS24_LOPO_stability_q30.{pdf,svg,png}` |

Steps 0, 2 and 3 run without raw EEG. Model: one-way repeated-measures ANOVA on participant × emotion means, Greenhouse–Geisser
corrected, Benjamini–Hochberg within topology (18), synchronization (6) and persistence (18) families in every sample
(`lopo_lib.py`). The earlier figure used q = 13 MTF/QG inputs and no sphericity correction; the paper uses q = 30.

```bash
python verify_original.py && python lopo_recompute.py && python make_figure.py
```
