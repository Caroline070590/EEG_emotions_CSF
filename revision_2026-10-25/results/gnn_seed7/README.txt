SEED-VII: participant-held-out EEG prediction

Generalization: unseen participants viewing familiar videos under the same session design.
Models are trained within SEED-VII; this is not transfer of FACED-trained model weights.
Primary results use seed 20261024; seeds 11, 29 are separate sensitivity analyses.
All three models receive the same 5,520 scalar inputs. No model ensemble is formed.

Balanced accuracy is reported in percent; differences and feature importance are percentage points.
There are 7 classes; the nominal equal-class level is 14.29%.
Inference uses model-specific permutation baselines rather than the nominal level.
95% intervals resample participants and retain all their trials. They condition on fitted OOF models
and this stimulus set; they are not intervals for generalization to new videos.

Permutation inference remains pending until every planned permutation completes for a model.
No p-value or claim of above-null prediction is generated from a partial permutation sample.
Holm adjustment across the prespecified models waits for all model nulls to complete.

Permutation units are 56 source-program-by-condition blocks.
Labels remain shared within each block and across participants; block-label counts are preserved.
Clip-label counts may change because source-condition blocks contain different numbers of clips.
The two mixed-condition sources (Under the Hawthorn Tree and You Are My Life) each contribute
separate sadness and happiness blocks, treated as separate permutation units. Dependence between
these blocks of the same source is not preserved by this null.
Conditions also differ in their session composition. The block null does not preserve
session-by-condition counts; above-null prediction alone does not establish session-independent
emotion information or generalization to unfamiliar stimuli.

Feature importance is the OOF balanced-accuracy drop after permuting a feature group across
videos within held-out participants, using a common video shuffle across participants. Repeats
are averaged within participant before bootstrapping. Positive values indicate model reliance;
negative values are retained. Correlated and overlapping groups are not additive, and importance
does not establish causation or a unique physiological contribution. Intervals are pointwise
descriptive intervals without multiplicity adjustment. Fold ranks are compared
separately within the atomic and family levels. Rank correlations are descriptive across five folds.

The overview's confusion matrix and feature panel use the prespecified hybrid GNN, not a model
chosen after inspecting accuracy. All models' confusion matrices and importance tables are retained.

This directory is an analysis report. No manuscript has been edited by this script.

Session-only balanced accuracy: 23.81%.
The dashed overview reference uses recording-session identity without EEG.
Paired EEG-minus-session-control differences are in session_control_paired_differences.csv.
This is a recording-design diagnostic, not a session-conditional permutation test.
