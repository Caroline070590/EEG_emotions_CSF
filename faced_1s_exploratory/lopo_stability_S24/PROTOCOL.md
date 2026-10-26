# Supplementary Fig. S24 — leave-one-participant-out (LOPO) stability: verification and recomputation

Written 26 October 2026, before any recomputed LOPO result was produced.

## Why

The S24 generator was never recovered (release README; `REVIEW_CHECKLIST.md`). The July 2026 numerical outputs survive in
`emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/LOSO_headline_effect_stability_{long,summary}.csv` with their input caches.

## Step 0 — original analysis identified (done)

`verify_original.py` reproduces all 42 full-sample and all 5,166 LOPO F statistics, p values and q values from the July caches
(maximum relative difference about 1e-11; q values to 1e-15). The original model is:

* outcome averaged within participant × emotion cells (9 emotions);
* participants with all 9 cells finite (complete cases);
* one-way repeated-measures ANOVA, F(8, 8(n − 1)), no sphericity correction;
* Benjamini–Hochberg within three domains per iteration: topology (3 representations × 6 metrics = 18),
  synchronization (6), higher-order topology (3 × 6 = 18);
* displayed: the 14 measures with full-sample q < 0.05; annotation = % of 123 LOPO iterations with q < 0.05.

Inputs that no longer match the current paper: July MTF/QG topology and MTF/QG persistence used q = 13 quantile states;
the paper uses q = 30. July synchronization inputs are byte-identical to the current corrected synchronization table
(global wPLI values unchanged by the estimator correction).

## Step 1 — current inputs

* Topology (Spearman, MTF q = 30, QG q = 30; 10% density): the retained trial-metric tables that generate main Fig. 1
  (`revision_2026-10-25/provenance/critical_review/figure1/src/cache/`).
* Synchronization: `faced_1s_exploratory/derived/trial_scalar_synchronization.csv` (3,444 rows, corrected wPLI).
* Persistence: recomputed from the 3,444 released 1-s segments with the unchanged release code at q = 30
  (`compute_persistence_q30.py`).

## Step 2 — model (fixed here)

Same cell construction, complete-case rule and families as the original. For each measure and each sample
(full and each of 123 LOPO samples):

* one-way repeated-measures ANOVA on the participant × emotion cell means;
* **primary p value: Greenhouse–Geisser corrected** (epsilon from the sample covariance of the 9 emotion columns),
  because sphericity across 9 levels cannot be assumed; the uncorrected p value (original model) is reported alongside;
* effect size: generalized eta squared;
* Benjamini–Hochberg within the three domains, separately within each sample.

Display: measures with full-sample primary q < 0.05. For each, the LOPO distribution of −log10(p_GG), its minimum
(worst single omission) and the % of 123 iterations with q < 0.05. A complete 42-measure table is also written, so the
selection is visible. This is a sensitivity analysis of the omnibus emotion effect; it carries no direction and does not
replace the emotion-versus-neutral contrasts or stimulus-level analyses.

## Decision rule for the manuscript

If the recomputed figure supports the existing wording, replace the figure and make the text exact. If some measures lose
support, report that explicitly. Remove S24 only if the result cannot be substantiated.
