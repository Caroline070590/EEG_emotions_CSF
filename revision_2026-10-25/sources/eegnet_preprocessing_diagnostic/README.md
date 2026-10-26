# FACED EEGNet reference-protocol audit

This isolated diagnostic examines whether preprocessing and scaling contribute to the low 10-s EEGNet scores in the existing matched experiment. It does not modify that experiment, reproduce the unpublished details of the 40.90% baseline, or provide an independent confirmatory evaluation. Its design follows inspection of the existing results, including results from participants reused here.

## Fixed design

All nine fits are specified before running this audit: three recipes, each with seeds 20261024, 11 and 29. The fixed zero-based participant IDs are 0–79 for training, 80–99 for validation, and 100–122 for test. Each 30-s trial supplies three disjoint 10-s windows. The primary score is window balanced accuracy. Trial balanced accuracy obtained by averaging the three probability vectors is secondary.

| Recipe | Resampling/filtering | Scaling |
|---|---|---|
| A: `window_filtered_zscore` | Independent 10-s windows; polyphase 250→200 Hz; fourth-order zero-phase 1–45 Hz Butterworth | Per-channel training-only mean and population SD |
| B: `trial_fourier_div100` | Fourier resample full 30-s trial to 6,000 samples; then split into three 2,000-sample windows; no added filter | Divide amplitudes by 100 |
| C: `trial_fourier_zscore` | Identical prepared EEG to B | Per-channel training-only mean and population SD |

B/C isolates scaling. A/C examines filtering and resampling together; it cannot distinguish the effects of filtering from resampling. A/B changes both processing and scaling. All recipes use the same 30 channels, architecture, optimizer, loss, validation criterion and training limits.

The model is a snapshot of our documented EEGNet-8,2 adaptation to 200 Hz, with kernels 100/25, dropout 0.5, average pools 4/8, and spatial/classifier max-norm constraints. Adam uses learning rate 0.001 and no weight decay or scheduler. Cross-entropy uses inverse training class frequencies. Maximum epochs=100, minimum before stopping=20, patience=20. Select the checkpoint by validation window balanced accuracy, breaking ties with lower unweighted validation cross-entropy. Do not refit on validation participants. Test prediction is performed once after selection; all nine results must be retained and reported. Test outputs must not determine changes to the recipes or selection of a supposedly reproduced baseline.

## Provenance and limits

The official BIOT repository at commit `d138e32634e52ae9fa6ec98ac9c4087b14ca869a` is MIT-licensed (copyright 2023 Chaoqi Yang). Its tracked source has no EEGNet module or FACED implementation, and text search found neither term. Therefore this audit does not claim to recover a BIOT EEGNet recipe. The local `provenance/BIOT_LICENSE.txt` records the license; no BIOT model code is executed or imported.

The official CBraMod FACED preprocessing at commit `b9e961003214326972c567eff390e75b0287e32a` uses the fixed 80/20/23 split, 32 channels, full-trial Fourier resampling and three 10-s windows. Its dataset loader divides amplitudes by 100. These data-processing choices motivate B; the public CBraMod optimizer and model defaults concern CBraMod itself and are not attributed to the missing EEGNet baseline recipe.

- [Pinned official FACED preprocessing](https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/preprocessing/preprocessing_faced.py)
- [Pinned official FACED loader](https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/datasets/faced_dataset.py)
- [Pinned BIOT model directory](https://github.com/ycq091044/BIOT/tree/d138e32634e52ae9fa6ec98ac9c4087b14ca869a/model)
- [EEGNet authors' architecture](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)

No original 32-channel FACED pickle files were found locally. The available cache contains 30 channels. TorchEEG's FACED loader defaults to the first 30 of the original 32 channels, dropping A1/A2; the export notebook used that default and removed its temporary original downloads. All recipes therefore use the existing 30-channel cache. Physical amplitude units have not been independently verified; `/100` follows the released loader literally. These facts prevent claiming exact reproduction of the published baseline. The model is trained from scratch; no pretrained weights are used.

The fixed test participants and videos have already appeared in previous experiments. This is a post-result diagnosis with validation-only fitting, not a fresh confirmatory benchmark. Results apply to new participants within this fixed familiar-video design, not unseen videos. No significance against a label-permutation baseline is claimed. Reported participant-bootstrap intervals are descriptive and conditional on the fitted models.

## Commands for review before launch

Use the existing cluster GPU Python environment. Input paths may point to the already-verified raw cache and its trial CSV. Initialization hashes all source files and inputs; later source changes invalidate that protocol.

```bash
python protocol.py --raw /path/cache/eeg_30s.npy --metadata-csv /path/cache/features_scalar.csv --out /path/audit_results
python prepare.py --config /path/audit_results/config.json
python run.py --config /path/audit_results/config.json --task 0 --device cuda
```

Tasks 0–2 are A, 3–5 B, and 6–8 C, in the seed order above. Run tasks 0–8 only after review. Preparation creates two float32 arrays of approximately 2.48 GB each; B/C share one array. GPU tasks can run independently after preparation completes. `--smoke --device cpu` uses one participant per role, one video per class, all three windows, and one epoch, writing separate non-complete outputs. It is never included in the report.

```bash
python test_audit.py
python report.py --config /path/audit_results/config.json
```

The report verifies prediction checksums, lists every available recipe/seed and explicitly marks pending fits. Paired descriptive comparisons are generated only after all nine fits finish. A `test_started.json` marker prevents silently repeating a test evaluation after a partial failure; incomplete output requires explicit review. No cluster submission is performed by these scripts.

After review, `submit_cluster.py` explicitly submits the bounded Slurm chain: CPU preparation (2 threads, 8 GB, 1 h), GPU smoke (2 threads, 12 GB, 1 h), nine GPU tasks with at most two concurrent (16 GB, 2 h each), then CPU report (8 GB, 1 h). It uses the existing EEG CPU/GPU environments and the verified cluster raw/CSV paths. The submission record is created exclusively before any job submission, updated after every stage, and blocks duplicate or silent partial resubmission. Only this explicit submission script submits jobs. TF32 is disabled, matching the existing canonical benchmark.
