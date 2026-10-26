# S24 leave-one-participant-out stability — results (26 October 2026)

Start with `PROTOCOL.md`. Run order: `verify_original.py` → `compute_persistence_q30.py` → `lopo_recompute.py` → `make_figure.py`.

## 1. Original analysis located and verified

* The generator script was never recovered, but the July 2026 outputs (`LOSO_headline_effect_stability_{long,summary}.csv`) and their
  input caches survive in `emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/`.
* `verify_original.py` reproduces all 5,166 rows (42 measures × 123 omissions) from those caches: F to 7e-11 relative, q to 6e-11
  (`results/ORIGINAL_REPRODUCTION.json`). The model was therefore a one-way repeated-measures ANOVA on participant × emotion
  means, without sphericity correction, BH within three families. The figure displayed the 14 measures with full-sample q < .05.
* **Problem found:** the July MTF/QG topology and MTF/QG persistence inputs used q = 13 quantile states; the paper uses q = 30.
  The original S24 therefore did not describe the current MTF/QG networks. Synchronization inputs are byte-identical to the
  current corrected table (wPLI values unchanged).

## 2. Recomputation on current inputs

* Topology: the q = 30 trial-metric tables that generate main Fig. 1. Synchronization: `derived/trial_scalar_synchronization.csv`.
  Persistence: recomputed from the 3,444 released 1-s segments (input contract passed) with the unchanged release code; Spearman
  persistence reproduces the July values exactly, MTF/QG now use 30 nodes.
* Model: as above, with Greenhouse–Geisser-corrected p values (primary) and generalized eta squared; BH within the same three
  families in every sample. Full-sample GG p values and F for all 42 measures independently match pingouin (`results/PINGOUIN_CHECK.json`).

## 3. Findings (`results/lopo_summary_all42.csv`)

* 21 of 42 measures have a corrected full-sample omnibus effect (previously 14 at q = 13 without sphericity correction).
  Newly included at q = 30: MTF γ, λ, path length, efficiency; QG transitivity, efficiency; MTF H1 entropy.
* 19 of 21 remain significant after correction in all 123 omissions, including all six synchronization measures.
* Spearman H1 total: 120/123 (97.6%). **QG transitivity: 88/123 (71.5%) — participant-sensitive.**
* Every displayed measure keeps nominal p < .05 in every omission (worst omission p = .030).
* Of the 21 measures without a full-sample effect, at most 8.1% of omissions produced a corrected effect.
* Effects are small (generalized eta squared 0.005–0.022). Videos are treated as fixed; this addresses participant dependence only.

## Decision

The S24 conclusion (principal effects not driven by a single participant) is substantiated on the current inputs, with
QG transitivity as an explicit exception. Replace figure and text with `manuscript_S24_replacement.tex` and
`figures/FigS24_LOPO_stability_q30.pdf`; removal is not needed. The main-text sentence on synchronization remains accurate.
