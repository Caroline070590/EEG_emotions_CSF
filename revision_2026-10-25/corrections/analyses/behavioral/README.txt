Behavioral prediction correction — local analysis, not yet published

The corrected Appendix J.4 analysis is in results/. No manuscript or original
analysis file was changed. The input is the validated September 4 verification
rerun, not claimed byte-identical to the earlier manuscript-linked outputs.
Its raw synchronization metrics are unchanged; the independent wPLI audit
verified all 3,444 trial values against standard amplitude-weighted wPLI.
The cached behavioral ratings match the official events to rounding precision.

Run from any working directory:
  python correct_prediction.py --out results_new
The default results/ directory is protected from overwrite. The seven regression
tests are in test_prediction.py. verify_saved_outputs.py independently checks
all predictions, fold scalers, original-unit inverse transformations, errors,
paired tests and corrections in the existing results/ directory.

Use results/Table_S9_corrected.tex as a proposed replacement for tab:s9-loov.
Panels A/B show errors in original/training-SD units; Panel C reports the 12
original-unit comparisons. Full errors and tests in both scales are in CSVs.
Neither scale yields a BH-significant comparison. The three accompanying
LaTeX/text snippets are proposals for root integration only.

Use results/Supplementary_Figure_S23_behavioral_EEG_models_corrected.pdf under
a NEW manuscript filename: its association panels retain historical Table S7
and S7b coefficients; panel c uses corrected prediction errors. The plotting
routine explicitly orders metrics. The separate original-unit figure is also
provided. Existing Figure S24/RSA and other generators are not certified by this
J4 correction. Nothing was uploaded or represented as publicly released.

Outcome scaling uses 27 training videos per participant, population SD, and a
scale of one original outcome unit when training variance is zero. This occurs
only for participant 23's constant Lag measure, in all 28 folds. No participant
is dropped. Familiar participants are shared across train/test; each video is
held out of fitting once. The paired fold tests are exploratory because their
training sets overlap. Correction changed outcome preprocessing, not the
scientific interpretation: no model contrast survives multiplicity correction.
