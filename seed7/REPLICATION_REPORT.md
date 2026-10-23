# SEED-VII replication vs FACED

Generated automatically by `src/figS8_compare_with_faced.py`. Plan: `ANALYSIS_PLAN.md`.

## 1-2. Emotion-vs-neutral tests that survive BH-FDR

| Dataset | Family | Tests | q<.05, stimuli fixed | q<.05, stimuli random (SEED-VII: clips / sources) |
|---|---|---|---|---|
| FACED | primary (topology + sync) | 224 | 79 | 0 |
| SEED-VII | primary (topology + sync) | 168 | 18 | 0 / 0.0 |
| FACED | recurrence (RQA + JRS) | 48 | 27 | 0 |
| SEED-VII | recurrence (RQA + JRS) | 36 | 27 | 23 / 23.0 |
| FACED | of which topology | 144 | 40 | 0 |
| SEED-VII | of which topology | 108 | 8 | 0 / 0.0 |
| FACED | of which synchronization | 80 | 39 | 0 |
| SEED-VII | of which synchronization | 60 | 10 | 0 / 0.0 |

## 3. Decoding on unseen stimuli

| Dataset | Folds | Features | Accuracy | Chance | Null mean | Gain | p |
|---|---|---|---|---|---|---|---|
| FACED | leave-one-video-out | Band-specific sync + power | 13.7% | 11.1% | 6.4% | +7.2 | 0.001 |
| FACED | leave-one-video-out | Spearman edges (signed) | 7.8% | 11.1% | 6.4% | +1.4 | 0.166 |
| FACED | leave-one-video-out | MTF q=30 edges | 6.5% | 11.1% | 6.5% | -0.0 | 0.480 |
| FACED | leave-one-video-out | QG q=30 edges | 6.6% | 11.1% | 6.7% | -0.1 | 0.562 |
| SEED-VII | leave-one-source-out | Band-specific sync + power | 29.5% | 14.3% | 12.3% | +17.2 | 0.001 |
| SEED-VII | leave-one-source-out | Spearman edges (signed) | 16.5% | 14.3% | 12.7% | +3.8 | 0.001 |
| SEED-VII | leave-one-source-out | MTF q=30 edges | 13.3% | 14.3% | 12.1% | +1.1 | 0.128 |
| SEED-VII | leave-one-source-out | QG q=30 edges | 11.6% | 14.3% | 12.5% | -0.9 | 0.806 |
| SEED-VII | leave-one-source-out | RQA + JRS | 22.7% | 14.3% | 12.4% | +10.3 | 0.001 |

## Direction check (anger, disgust, fear, sadness; identical outcomes)

* 136 outcome x emotion pairs; Spearman r of d_z = 0.17 (p = 0.0539); same sign in 58%.
* Pairs significant with stimuli fixed in both datasets: 11; same sign in 100%.

## Reading guide

* FACED conclusion 1 replicates if SEED-VII, like FACED, has many fewer surviving tests with stimuli random than fixed.
* Conclusion 3 replicates if band-specific synchronization + power decodes above its own null on held-out sources.
* SEED-VII differs from FACED in culture, lab, montage, 7 vs 9 conditions, and 20 vs 123 participants (fewer
  participants lowers power with stimuli fixed; 12 vs 3 clips per emotion raises it with stimuli random).
