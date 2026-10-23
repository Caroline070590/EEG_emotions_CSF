# Pre-specified model-selection rule (written 13 Oct 2026, before any real-data GNN run)

1. Hyper-parameters (epochs = 40, hidden = 8, code = 16, dropout = 0.3, lr = 3e-3,
   weight decay = 1e-3, batch = 128) are fixed and not tuned on real data.
2. lambda_adv for the real-data analysis = the value in {0, 0.3, 1.0} with the highest mean
   leave-one-video-out balanced accuracy across all simulations with delta > 0, provided its
   mean accuracy in the delta = 0 null simulations does not exceed that of the other models
   (i.e., no inflated false-positive decoding).
3. Real data are then analysed once with that setting (plus the lambda = 0 ablation and the
   LDA/MLP baselines), with 1,000 stimulus-structured permutations for the chosen model.
4. All runs are reported, including any in which SD-GNN does not outperform the baselines.

## Amendment (13 Oct 2026, still before any real-data GNN run)
A constrained GNN variant (`cgnn`: independent sigmoid view gates in [0, 1] with a fixed L1
shrinkage of 0.01 on the gates; otherwise identical) was added to the simulation study at the
author's request, with lambda_adv in {0, 1.0}. The selection rule above is applied to the
union of SD-GNN and cgnn settings: the single (model, lambda) with the highest mean simulated
accuracy over delta > 0, subject to the same null check, is analysed on the real data.

## Amendment 2 (13 Oct 2026, still before any real-data GNN run)
The first simulation round was uninformative: with sparse planted effects (20 of 4,935 features)
and video fingerprints 10-20x larger than in FACED, no method (including LDA) detected the
effect. The simulation was therefore recalibrated to FACED (`--data simcal`): planted effects
are scale x the real condition-mean patterns (scale in {0, 0.5, 1}); video fingerprints use the
real between-video-within-condition SD per feature x kappa (kappa in {1, 3}). Because every
leave-one-video-out classifier is biased below nominal chance in this design, the selection
criterion is each method's mean GAIN over its own scale = 0 null (same kappa and seed),
averaged over scale > 0; the null check is unchanged.

## Real-data run (13 Oct 2026)
Selected by the rule: cgnn with lambda_adv = 1.0 (highest mean simulated gain among the GNN
settings; null check passed). In the calibrated simulations, MLP and LDA showed about three
times larger gains than every GNN setting. The real data are analysed once with all seven
settings (observed labels), and with 200 stimulus-structured permutations each for the
selected cgnn, MLP, and LDA. All results are reported.

## Amendment 3 (14 Oct 2026): hybrid model and recurrence features, before any hybrid real-data run
Motivation (from the calibrated simulations only): message passing compressed away the dense,
global coupling shifts that carry the condition signal. The hybrid model adds a direct path for
global summaries (mean edge weight of every view, channel-mean of every node feature) to the
selected constrained GNN with the video adversary (lambda = 1.0). No feature is selected using
condition statistics, so there is no circularity.
Recurrence features (RQA node features, joint-recurrence synchronization view) use fixed,
pre-specified parameters (see src/02_recurrence_features.py).
Pre-specified primary real-data model: hybrid, lambda = 1.0, with recurrence features.
Reported alongside: hybrid lambda = 0 and hybrid without recurrence (ablations); shrinkage LDA
on the global summaries only (with and without recurrence); and the earlier cgnn lambda = 1.0
with recurrence. 200 stimulus-structured permutations for the primary hybrid and for the
summary-only LDA with recurrence. The primary comparison is gain over each model's own null.

## Amendment 4 (14 Oct 2026): summary-first model, cross-video invariance, realistic simulation
Written before any hybrid real-data result was inspected (the hybrid real-data array was still
running; its outputs were not read). Motivation, from simulations only: in the calibrated
simulation, shrinkage LDA on the 16 global summaries outperformed every network, and inside the
hybrid the summary path was dominated by the larger graph path.
1. Model `sfgnn` (summary-first): logits = linear head on the batch-normalised global summaries
   + alpha x (constrained-GNN head), with a scalar alpha initialised at 0 and an L1 penalty of
   0.01 on |alpha| (same fixed shrinkage as the view gates). The GNN can only add what the
   summaries miss; alpha -> 0 means no spatial structure beyond the global shift.
2. Cross-video invariance (V-REx, Krueger et al. 2021): loss = mean over training videos in the
   batch of the per-video loss + beta x their variance, beta = 1.0 fixed (not tuned).
   Variants: sfgnn lambda_adv = 0; sfgnn lambda_adv = 1.0; sfgnn + V-REx (lambda_adv = 0).
3. Realistic simulation (`--data simreal`): as `simcal`, but (i) video fingerprints are drawn
   with the real between-video-within-condition covariance (from the 28 real video-mean
   deviations, eigenvalues corrected for sampling noise 1/123), so fingerprints are correlated
   across features; and (ii) trial noise is resampled from real trial residuals (real trial minus
   its video mean), so noise is correlated across features. This removes the simulation's
   advantage for averaging. Settings: scale {0, 0.5, 1} x kappa {1, 3} x 3 seeds; models lda,
   lda_summary, cgnn 1.0, hybrid 1.0 and the three sfgnn variants.
4. Selection: the sfgnn variant with the highest mean gain over its own null in `simreal`
   (averaged over scale > 0, null check as before) is the primary Amendment-4 real-data model.
   All three variants are run on the real data with recurrence features (observed + 200
   stimulus-structured permutations each), and all are reported.

## Amendment 5 (14 Oct 2026): biology-informed graph prior, before any run
Suggested by a colleague. Question: does a graph built from biological priors help, compared
with the same network without a graph and with a physics-informed (scalp-distance) graph?
- Network: constrained GNN (`cgnn`, sigmoid gate, fixed L1 shrinkage 0.01), lambda_adv = 0, with
  node features = log band power (5) + channel-wise RQA (5), standardized within participant.
  The data-derived edge views are replaced by ONE fixed adjacency (`--prior`):
    none     : no edges (message passing carries no neighbor information)
    physics  : exp(-(d/median d)^2) of inter-electrode distance, MNE standard_1020 template
    biology  : edge (weight 1) between electrodes of the same cortical region (frontal, central,
               temporal, parietal, occipital; the paper's regional grouping) and between the 13
               left-right homologous pairs (callosal interhemispheric connections)
  Each adjacency is symmetrically degree-normalized. Everything else identical.
- Evaluation: leave-one-video-out, observed + 200 stimulus-structured permutations for each prior
  (same permutation indices across priors, so differences are paired).
- Primary contrast: biology minus none (one-sided gain test against the paired permutation
  differences). Secondary: each prior's gain over its own null; physics minus none.
- Reported in full whatever the outcome.
Note (added after a 3-epoch smoke test, before the full run; design unchanged): with the paper's
regional grouping, all 13 homologous pairs lie within the same region, so the biology graph equals
the within-region graph (89 edges). The smoke test (3 epochs, one permutation) only verified that
the code runs and that permuted labels give null-level accuracy (10.7%); it is not a result.

## Amendment 6 (14 Oct 2026): biology-masked synchronization GNN, before any run
Requested after discussing Amendment 5 with a colleague ("biology-informed GNN" = anatomy as a
mask on the measured synchronization). Amendment-5 results were seen only as interim numbers
(no graph 17.1%, physics 17.0%, biology 17.2%); this design does not depend on them.
- Network: constrained GNN (`cgnn`, sigmoid view gates, fixed L1 0.01), lambda_adv = 0; node
  features = log band power (5) + channel-wise RQA (5); edge views = the 10 measured
  synchronization views only: wPLI and oAEC in delta, theta, alpha, beta, gamma (standardized
  within participant, as before).
- `--sync-mask`:
    full    : all 435 edges of every view (no mask)
    biology : edges kept only between electrodes of the same cortical region (the paper's regional
              grouping; the 13 homologous pairs are within regions), 89 edges, same mask for all views
    random  : 89 edges drawn uniformly at random (fixed seed 2026), same mask for all views -
              controls for sparsity
  Masked edges are set to 0 (= the participant mean after within-participant standardization).
- Evaluation: leave-one-video-out, observed + 200 stimulus-structured permutations per mask, same
  permutation indices across masks (paired).
- Primary contrast: biology minus random (does anatomy matter beyond sparsity?). Secondary:
  biology minus full; each mask's gain over its own null. One-sided gain tests. All reported.
