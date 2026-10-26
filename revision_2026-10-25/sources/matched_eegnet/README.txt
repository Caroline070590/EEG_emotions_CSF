Matched EEGNet augmentation experiment — protocol fixed 24 October 2026

Question
Does recurrence, connectivity, or their combination improve an EEGNet baseline
when all models receive the same EEG, participants, training splits and evaluation?
This experiment assesses feature augmentation of EEGNet. It is not a multiplex
GNN, a replication of a published benchmark score, or evidence of improvement
until the held-out comparisons are complete.

Design
FACED: 123 participants, 28 videos, nine classes.
SEED-VII: 20 participants, 80 clips, seven classes.
Fit separate models for each dataset; no cross-dataset training or transfer.
Five participant-held-out folds match the earlier participant evaluation.
Within each outer training set, hold out 20% of participants for epoch selection.
Then reinitialize and refit on all outer-training participants for the selected
number of epochs. The outer test participants never select epochs or settings.
All models share folds and training seeds. Primary seed: 20261024; sensitivity
seeds: 11 and 29. All three seeds will be reported, regardless of outcome.

Input windows
Each original 30-second trial is analyzed either as one 30-second window or as
three nonoverlapping 10-second windows. Split first, resample 250 to 200 Hz,
then apply a fourth-order 1–45 Hz zero-phase Butterworth filter independently
within each window. The exact returned float32 signal supplies both EEGNet and
the feature extractor. Features from a 30-second trial are never reused for
its 10-second inputs. Train-only channel scaling supplies EEGNet; train-only
column scaling supplies the side features. Constant training columns are zeroed.

Primary predictions are at the participant–video trial level. For the 10-second
model, average the three class-probability vectors before scoring. Both duration
conditions therefore use 30 seconds of EEG per primary prediction. Individual
10-second-window results are descriptive, not independent participant samples.
Duration also changes EEGNet's flattened classifier dimension; the duration
comparison evaluates the complete windowing design, not duration alone.

Models
1. EEGNet-8,2 alone.
2. EEGNet plus connectivity: five wPLI views, five pairwise-orthogonalized AEC
   views and signed Spearman correlation; 4,785 upper-triangle edge values.
3. EEGNet plus recurrence: five RQA values per channel plus 435 joint-recurrence
   values; 585 features.
4. EEGNet plus both feature families: 5,370 features.
5. Identical combined model with the auxiliary vectors shuffled jointly across
   trials independently within each participant. Preserve window position and
   keep a trial's three donor windows together. Reshuffle training each epoch;
   validation/test donor maps are fixed and independent. Labels never select
   donor mappings. This control assesses the usefulness of trial-aligned side
   information beyond the combined branch's additional parameter capacity.

RQA uses fixed two-second subwindows in BOTH duration conditions (five or fifteen
blocks), resampled from the common signal to 125 Hz: embedding dimension 3,
delay 3 samples, target recurrence rate .05, Theiler exclusion 2, minimum line
length 2. Average the block descriptors. These two-second blocks are a new
matched-window setting, distinct from the earlier three-second RQA analysis.
Threshold ties are retained; actual recurrence rate can exceed .05 for tied
distances. Connectivity and recurrence branches exclude the optional logpower
features returned by the extractor.

EEGNet follows the authors' EEGNet-8,2 implementation with temporal kernel 100
and separable kernel 25, explicitly adapted to 200 Hz; pooling 4 then 8, ELU,
dropout .5, spatial max-norm 1 and classifier max-norm .25. The side branch is
input -> 32 units -> class logits, with ELU/dropout .5 and no biases. Its final
matrix starts at zero; all variants initially reproduce the same EEGNet logits.
The side logits are added to the raw EEGNet logits. The additional branch can
have substantially more parameters than EEGNet, motivating the shuffled control.
Reference: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py

Training
Adam, learning rate .001, batch 64, inverse-frequency class-weighted cross-entropy
computed only from training labels. Maximum 100 selection epochs; run at least
20, then stop after 20 epochs without validation improvement. Select the best
trial balanced accuracy, with lower unweighted trial cross-entropy breaking ties.
An early epoch may be selected even though the selection phase runs at least 20.
Refit from fresh identical initialization for the selected epoch count.
Store validation curves, exact participant IDs, scalers, selected epoch,
donor maps, checkpoints and all out-of-fold class probabilities.

Comparisons and interpretation
Primary contrasts for each dataset and duration: connectivity minus baseline,
recurrence minus baseline, combined minus baseline, combined minus shuffled.
This gives 16 contrasts. Report gains in percentage points, 2,000 participant
bootstrap intervals, and 9,999 two-sided participant sign flips with +1 correction.
Apply Holm across all 16 primary contrasts only when the family is complete.
These paired tests are approximate and conditional on the fitted out-of-fold
models and the familiar stimulus set. They assume exchangeability/symmetry of
participant differences; overlapping training sets and training instability are
not fully represented. They are not full-refit label permutations or a test of
performance against random labels. Seed sensitivity is descriptive and does
not create additional independent participants or change the primary seed.
Generalization is to new participants viewing these video stimuli. Separate
within-dataset SEED-VII training is not cross-dataset external validation.
SEED-VII also includes a train-fold, class-balanced session-only diagnostic using
no EEG. Its condition imbalance across sessions can yield above-chance scores.
Report every model's paired difference from that control, without adding a new
P-value family. Beating a nominal chance value alone does not establish information
beyond this recording-design association.
No model settings or manuscript claims are selected using outer test results.

Execution
Local software tests:
  python -m unittest discover -s matched_eegnet_2026-10-24 -p 'test_*.py' -v

The cluster project runs under ~/EEG_emotion_revision_2026-10/.
Prepared signals and the SEED raw copy live under
  ~/scratch/eeg_matched_2026-10-24/
The isolated CUDA environment leaves the existing CPU analysis environment intact.
submit_cluster.py submits provenance/software preflight, a synthetic GPU probe,
256 feature chunks, four merges, configuration freeze, a real-data software smoke,
300 GPU fold jobs (maximum eight concurrent), and a dependent final report.
Source and input hashes are recorded; completed tasks are reusable only under
the same frozen configuration. A failed prerequisite prevents dependent stages.
CLUSTER_JOBS.json records every submitted job; inspect it before any resubmission.

Report outputs
results/report/ contains trial/window scores, paired contrasts and adjusted
P values, seed sensitivity, descriptive duration comparisons, confusion matrices
and figures. Incomplete folds remain pending and do not yield aggregate claims.
The manuscript and reviewer response are unchanged until results are verified.
