# Independent replication in SEED-VII

Section 3.2 and Appendix C of the manuscript (Fig. S8). The main FACED analyses are repeated on SEED-VII
(20 participants; 80 clips: 12 per non-neutral emotion and 8 neutral) with the same feature code, following
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md), which was written before the data were analyzed and logs every later
addition.

## Pipeline (`src/`)

| Step | Script | Output (`tables/`) |
|---|---|---|
| 0 | `00_load_clips.py` | last 30 s of each clip, the 30 FACED channels, resampled 200 → 250 Hz |
| 1 | `01_trial_features.py` | synchronization and topology (primary family) |
| 2 | `02_recurrence_features.py` | recurrence quantification |
| 3 | `03_stimulus_aware_contrasts.py` | `all_contrasts.csv`, `omnibus_tests.csv`, `summary.csv` |
| 4 | `04_source_nested_contrasts.py` | `contrasts_source_nested.csv` (source programmes as stimulus unit) |
| 5 | `05_decoding_unseen_sources.py` | `decoding_unseen_sources.csv`, `decoding_leave_one_clip_out.csv` |
| 6 | `06_decoding_source_block_null.py` | `decoding_source_block_null.csv` |
| 7 | `07_normalization_sensitivity.py` | `normalization_sensitivity.csv` |
| 8 | `08_recurrence_delta_adjustment.py` | `recurrence_delta_adjustment.csv` (exploratory) |
| | `figS8_compare_with_faced.py` | Fig. S8, `comparison_with_faced_*.csv`, `REPLICATION_REPORT.md` |

## Running

The SEED-VII data are available on request from the BCMI laboratory and may not be redistributed. With the
official release (`EEG_preprocessed/*.mat` and the metadata spreadsheets) at `/path/to/SEED-VII`:

```bash
./run_all.sh /path/to/SEED-VII      # about 3 hours on 12 cores
```
