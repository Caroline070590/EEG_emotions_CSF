# Synchronization estimator and coupling provenance

The retained one-second synchronization values use standard amplitude-weighted wPLI. The public global synchronization source had regressed to a unit-phase calculation; its isolated replacement now retains the analytic amplitudes and rejects caches without matching estimator, source, and output checksums. The raw-input checksum manifest is separate: cache reuse itself does not detect a changed raw-input directory.

Independent recomputation covered all 3,444 one-second trials. Retained global wPLI agrees to a maximum absolute difference of 6.11e-8; the behavioral/stimulus-aware table agrees with the independent SOS implementation to 4.68e-8. The historical phase-only cache is demonstrably different (mean absolute difference 0.0827). Therefore the retained standard-wPLI results do not require replacement by a different numerical estimator. The incorrect public source does require correction.

## Exact displayed models

Main Figure 2 is reconstructed from `emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/cache/trial_synchrony_metrics.csv` and `trial_metrics_spearman.csv`. These are merged one-to-one on participant, emotion, and trial. Each variable is standardized within participant across the 28 trials, using the sample standard deviation plus 1e-9. There are 3,444 rows and 123 participant clusters.

- Pooled coupling: OLS `topology_z ~ synchronization_z`, with an intercept and participant-clustered covariance. It has no emotion fixed effects.
- Emotion-resolved coupling: OLS `topology_z ~ synchronization_z * C(emotion)`, neutral reference, with participant-clustered covariance. Each condition slope is the appropriate linear combination. The interaction is a joint Wald test of the eight interaction coefficients. It is not a likelihood-ratio test.
- All 24 pooled and all 216 emotion-resolved fitted coefficients reproduce the colors of the retained figure exactly. All 54 synchronization fingerprint cells also reproduce exactly. A raster-color match validates a displayed color bin, not otherwise unavailable hidden floating-point coefficients. Recovered exact coefficients are provided separately.
- The nine condition-specific means of absolute coefficients and nominal-significance counts agree with the original bar-panel annotations. Coupling p values are nominal and unadjusted; they do not establish corrected emotion differences.

The later released scaffold instead described and attempted a 1,107 participant–emotion MixedLM. It does not reproduce this figure: the attempted pooled fits were singular. That unsuccessful check is retained in `GLOBAL_FIGURE_VALIDATION.json`; it is not the successful historical validation in `HISTORICAL_GLOBAL_FIGURE_VALIDATION.json`. The repaired public generator restores the proven historical OLS computation, rather than silently substituting another analysis.

## Distinct sensitivity analyses

| Analysis | Observations and estimator | What it supports |
|---|---|---|
| Global coupling, main Fig. 2 | 3,444 matched trials; within-participant standardized clustered OLS | Descriptive synchronization–topology associations |
| Regional coupling, main Fig. 3 | 3,444 matched trials per region; within-participant standardized nodal strength on synchronization | Positive within-sensor-group association |
| Regional split halves, S19 | Same trial-level slope, re-estimated in 300 complementary participant partitions | Direction preserved in both halves for all 30 region-by-measure pairs; not independent replications |
| Participant averaging, S23 | 1,107 participant–emotion cells; omnibus emotion tests | Sensitivity of emotion effects to averaging; not a coupling-slope sensitivity |
| Headline LOPO, S24 | Historical omnibus emotion-effect figure; exact generator unrecovered | Historical figure only; not independently certified here |
| Synchronization split halves, S25 | Nine condition-versus-neutral contrasts | Directional stability of contrasts, not of global coupling |
| Thirty-second coupling nulls | Separate long-window data and model | Supplementary validity analysis; not evidence that the one-second figure used 1,107 MixedLM observations |

All 30 regional full-sample coefficients and the 300-partition directional-stability results reproduce to numerical precision. The plotted regional wPLI means, strength ranges, and rounded coefficients remain unchanged. The full 48 synchronization contrast tests retain exactly the same nine FDR discoveries; no wPLI contrast survives that family. Small differences in some non-wPLI tests are explicitly recorded in `DOWNSTREAM_VALIDATION.json`, rather than represented as byte-identical reproduction.

All 336 band-standardized effects are reconstructed from the retained cache. Raw wPLI verification sampled 63 trials across all seven bands (441 estimates), with the largest discrepancy 7.41e-6 in delta. It did not rerun every narrow-band raw trial or certify every omnibus p value. The 1–45 versus 4–45 Hz effect correlation is checked separately in `BAND_EFFECT_VALIDATION.json`.

Bootstrap effect-size comparisons (eight wPLI-versus-neutral and all 36 wPLI pairwise comparisons), behavioral inputs, and stimulus-aware inputs have verified standard-wPLI lineage. No claim is made that every historical figure generator has been recovered. In particular, the missing headline-LOPO generator for current S24 is different from the recovered behavioral-RSA generator whose historical filename contains S24 but is current S29.

The focused regression suite checks analytic amplitude weighting, stale-cache rejection, and all 240 recovered coupling coefficients. The original public release and original manuscript deliveries remain unchanged.
