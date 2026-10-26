Matched EEGNet benchmark

Matched EEGNet augmentation comparison in unseen participants viewing familiar stimuli

New participants, familiar videos and sessions; models trained within each dataset. Not unseen-stimulus generalization or transfer of FACED-trained weights to SEED-VII.

Three nonoverlapping 10-s windows are probability-averaged per trial and compared with one 30-s window. Both designs use 30 s; individual-window accuracies are descriptive.

Epoch selection uses participant-disjoint inner validation and trial-level balanced accuracy, with validation loss breaking ties.
Each outer model is then refitted from fresh weights on all outer training participants for the selected epoch count.
Feature normalization uses training participants only. Connectivity and recurrence use the same input window as EEGNet.
The combined and shuffled-combined models have identical input dimensions and capacity; their auxiliary features are jointly shuffled within participant.

Paired participant bootstrap retaining all familiar videos/windows; conditional on fitted models and stimulus set. Pointwise 95% intervals; not simultaneous intervals.

Two-sided participant sign flips assume exchangeability of paired model scores; conditional on fitted OOF models. Overlapping training sets make this an approximation; it excludes training-instability uncertainty. These are not full-refit label-permutation tests and do not establish performance above a stimulus-structured null.

One Holm family of 16 primary-seed comparisons: three augmentations versus EEGNet and combined versus shuffled-combined, across two datasets and two window designs. Holm values withheld until every comparison is complete. Sensitivity seeds and duration comparisons are descriptive.

Augmentations are engineered descriptions of the same EEG window; improvement does not establish independent physiological information or causal relevance.

The SEED-VII session-only diagnostic uses no EEG and fits the class-balanced session decision from outer-training participants. The same trial predictions apply to both window designs. Merely exceeding nominal chance or this control does not establish session-independent emotion information; control comparisons have descriptive intervals only.

Missing folds remain pending: partial OOF predictions do not generate model scores or comparative inference.
Positive paired differences favor the first named model; significant negative differences do not support an improvement.
The report does not modify manuscript files.
