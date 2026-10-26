VERIFIED ANALYSIS RESULTS — 25 OCTOBER 2026

matched_eegnet: all 300 participant-held-out folds, all 20 primary model/context
combinations and both sensitivity seeds. The report includes every planned model,
the 16 Holm-adjusted primary comparisons, confusion matrices and session controls.

seed_stability: every seed score and paired gain, with all 300 fold diagnostics.
Spread across seeds is descriptive. No seed was selected from its test score.

gnn_faced and gnn_seed7: completed input-matched participant analyses, including
all 99 full-refit permutations per model, feature ranks and confusion matrices.
These concern familiar stimuli in new participants and do not replace the older
held-video/source analyses in the manuscript.

All 10-s primary scores average three window probabilities from the same 30-s
trial. Both window designs therefore use 30 seconds per primary prediction.
SEED-VII's session-only control must accompany its participant-held-out scores.
Bootstrap intervals and sign-flip tests condition on fitted models and sampled
stimuli; the latter are not full-refit label-permutation tests.

The completed nine-fit preprocessing audit remains a diagnostic, not a replication
of the published 40.90% EEGNet benchmark. Its brief disclosure and all nine scores
appear in Appendix B. The full diagnostic artifacts are stored separately in the
final review delivery under diagnostics/eegnet_reference_audit/.

frozen_sources contains the matched-EEGNet and preprocessing-diagnostic deployment
source archives, manifests and configurations. The FACED and SEED-VII participant
GNN source bundles are newly reconstructed from the exact deployment-listed files;
every member matches its recorded deployment hash. They are not represented as
original archive bytes. The original deployment manifests and configurations are
included, with provenance in ADDITIONAL_SOURCE_PROVENANCE.json. These additions
contain code and provenance only; the original EEG inputs are not included.

The files in this directory contain analyses, not a submission-ready manuscript.
See the separate manuscript and reviewer-response PDFs for proposed text changes.
