# SEED-VII independent replication: pre-specified plan

Written on 2026-10-22, before any SEED-VII EEG data were loaded or analysed.

## Purpose
To test whether the FACED conclusions hold in an independent dataset with a different lab, cohort,
culture, montage, amplifier and stimulus set (SEED-VII; Jiang et al., IEEE Trans. Affect. Comput.).
The FACED conclusions under test:
1. With participants AND videos as random factors, few or no emotion-vs-neutral differences in
   network topology (Spearman, MTF, QG; q = 30, density 10%) and band-specific synchronization
   (wPLI, oAEC; delta-gamma) survive BH-FDR, whereas many survive when videos are treated as fixed.
2. Recurrence (RQA: DET, LAM, Lmean, ENTR, TT) and joint-recurrence synchronization show the same
   pattern (participant-only effects that do not survive with videos random).
3. Condition decoding that generalizes to unseen stimuli is above a stimulus-structured null but
   small in absolute terms.

## Data (no tuning on SEED-VII)
* Official preprocessed EEG (`EEG_preprocessed/*.mat`; 20 subjects, 80 clips, 62 channels, 200 Hz).
* The same 30 channels as FACED (all present in the SEED-VII 62-channel montage).
* The last 30 s of each clip (as in FACED), resampled 200 -> 250 Hz (polyphase, 5/4), giving the same
  30 x 7500 trial as FACED, so all feature code runs unchanged.
* Labels from `emotion_label_and_stimuli_order.xlsx`: 7 conditions (happy, sad, fear, disgust,
  anger, surprise, neutral); 12 clips per non-neutral emotion and 8 neutral clips.

## Features (identical code and parameters to FACED)
Primary family only: topology of Spearman, MTF, QG at q = 30, density 10% (6 metrics each) and
wPLI / oAEC in 5 bands, i.e. 18 + 10 = 28 outcomes x 6 emotion-vs-neutral contrasts = 168 tests,
BH-FDR once. Recurrence family: 5 RQA + JRS = 6 outcomes x 6 contrasts = 36 tests, BH-FDR once.
The q x density sweeps, surrogates and persistent homology are not repeated (sensitivity analyses).

## Inference
* Primary: min F' (Clark, 1973) with participants and clips as crossed random factors
  (F1: paired test over 20 participants; F2: clip means, error pooled within condition,
  df2 = 80 - 7 = 73). Secondary: participant-only test (clips fixed).
* Stimulus identity caveat (decided before analysis): some clips are repeated presentations of the
  same stimulus (e.g. clips 8 and 58), and many come from the same source film. Sensitivity analysis:
  min F' with the F2 unit = source program (clips averaged within source; sources spanning two
  conditions are split by condition).

## Decoding
* Shrinkage LDA, features z-scored within participant, balanced accuracy, chance 1/7.
* Primary folds: leave-one-SOURCE-out (all clips from the same film/program held out together), so
  no excerpt of a test stimulus is ever in training. Secondary: leave-one-clip-out.
* Null: 1000 permutations of the clip -> condition map preserving the class counts (12/12/.../8).
* Feature sets as in FACED: band-specific sync + power, Spearman edges, MTF edges, QG edges,
  and RQA + JRS.

## Success criteria
Conclusion 1 replicates if the number of primary tests with q < .05 drops substantially when clips are
random (relative to clips fixed). Conclusions 2 and 3 are assessed the same way. Results are reported
whatever they are.

## Deviation log (recorded as it happened)
* 2026-10-22 21:40, before the leave-one-clip-out results were computed: decoding took ~50 min per
  feature set locally (Yggdrasil unavailable). The primary leave-one-source-out analysis was completed
  with 1,000 permutations as planned; the secondary leave-one-clip-out analysis was rerun separately
  with 200 permutations (p resolution 0.005) and is reported in the repository only.
* Added after the pre-specified results were seen (exploratory): adjustment of the recurrence
  measures for relative delta power (`src/08_recurrence_delta_adjustment.py`).
* 2026-10-23, after an internal review and after the primary results were seen: a stricter decoding
  null that permutes condition labels of whole source-by-condition blocks (56 blocks), so that excerpts
  of the same programme always share a label (`src/06_decoding_source_block_null.py`).
* 2026-10-24, after a further internal review: (i) the relative delta power covariate was corrected to the
  bandwidth-weighted integrated delta fraction (the earlier version divided unweighted band means);
  (ii) a source-level sensitivity analysis strictly nested within condition (52 programmes, equal weighting)
  was added (`src/04_source_nested_contrasts.py`); (iii) the 56-block permutation null permutes the
  source-by-condition blocks independently (the two programmes spanning two conditions contribute two blocks).
