# Exploratory recurrence multiplex experiment

Status: implemented; real-data extraction and functional tests performed locally. Full 123-participant inference and Baobab submission are pending cluster connection details. No evidence of an accuracy improvement is claimed.

Question: does sensor-level recurrence information improve emotion-category prediction on unseen videos beyond correlation and synchronization?

Every layer has the same 30 sensor nodes. Edges are absolute 4–45 Hz Spearman correlation, alpha-band wPLI, and Jaccard similarity of within-sensor recurrence patterns (joint recurrence overlap). This third layer is a sensor recurrence-similarity network, **not** the conventional time-point recurrence graph. Sensor node features optionally include recurrence rate, determinism, laminarity, mean/max diagonal length and diagonal-length entropy. Recurrence rate uses the entire off-diagonal denominator; its target threshold rate uses only Theiler-eligible pairs. Line lengths are in 40-ms anchor steps. They depend on the stated embedding and exclusion settings.

Independently filter the two raw source blocks [0,3500) and [4000,7500), retain [500,3000) and [4500,7000) at 250 Hz. Each window is modeled separately; average predicted probabilities across windows and three seeds. Within a window, connectivity and recurrence use the same retained samples: this experiment does not replace the prior separate-window coupling analysis.

Recurrence: antialias resampling to 125 Hz, per-trial/channel normalization, dimension 3, delay 16 ms, anchors every 40 ms, exclude pairs within 96 ms, threshold fifth percentile of eligible distances, retaining ties. These fixed exploratory choices were specified before prediction. Sensitivity to other recurrence rates/delays remains untested.

Two graph-convolution steps with 16 hidden units and sensor-aligned interlayer mixing, followed by mean sensor pooling and a classifier. Four ablations: connectivity, recurrence+RQA, connectivity+RQA, full multiplex+RQA. Training-only RQA scaling, 40 fixed epochs, seeds 11/29/47. No outcome-based epoch/model selection. Three video folds cover each trial once; participants overlap folds, so this is not unseen-participant generalization. All 3,444 unique participant/video keys are required for the full run.

The 99-permutation pilot shuffles category assignments at whole-video level, preserves class counts, rebuilds stratified folds, and refits every model and seed. Primary contrast is full multiplex minus connectivity balanced accuracy. Report all four model null tests and the increment, with Holm adjustment over five tests. This is a global label-shuffle baseline, not a conditional independence test. A 99-permutation pilot has coarse p-value resolution (0.01); any later expansion must be reported and must not be based solely on obtaining significance. No independent test set remains untouched by the broader project’s exploratory model development.

## Baobab

Requires Python with numpy, pandas, scipy, scikit-learn, statsmodels, torch, joblib, networkx and gudhi (latter dependencies inherited from existing modules). Transfer this package plus core/analysis_core.py, multiplex_prediction/{experiment,model}.py, inputs/disjoint_signed/input_qc.csv and EEG-data-raw-multiscale/full_trial, preserving relative paths. Extraction validates source SHA256 and trial identity.

Set EEG_PROJECT to the remote workspace and EEG_PYTHON to the environment executable. Submit extract.slurm; only after all 48 tasks succeed, submit train.slurm with an afterok dependency. GPU partition/resource defaults follow UNIGE documentation but must be checked against the actual account. The array is capped at four concurrent GPU jobs. No job has been submitted yet.

After all 100 training tasks finish:

```bash
python recurrence_multiplex/summarize.py recurrence_multiplex/pilot99
```

Smoke predictions from one participant and one epoch are strictly functional checks, never manuscript results. Original manuscript and analysis packages remain unchanged.

Sources: https://arxiv.org/abs/0908.3447 (recurrence networks; established methodology), https://doc.eresearch.unige.ch/hpc/slurm (cluster scheduling).
