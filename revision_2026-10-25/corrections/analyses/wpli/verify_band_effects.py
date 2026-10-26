"""Reconstruct all reported band effects from the raw-sampled standard-wPLI cache."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/band_robustness_results_and_code/emotion_eeg_paper_fixed'

def main():
    cache = BASE / 'cache/trial_synchrony_bands.csv'
    table = BASE / 'supplementary_tables/TableS_band_robustness.csv'
    d, reported = pd.read_csv(cache), pd.read_csv(table)
    rows = []
    for row in reported.itertuples():
        column = row.band + '__' + row.metric
        means = d.groupby(['subject', 'emotion'])[column].mean().unstack()
        difference = means[row.emotion] - means.neutral
        effect = difference.mean() / pd.concat([means[row.emotion], means.neutral]).std()
        rows.append(dict(band=row.band, metric=row.metric, emotion=row.emotion,
                         standardized_effect=effect, published_effect=row.standardized_effect,
                         difference=effect-row.standardized_effect))
    results = pd.DataFrame(rows)
    results.to_csv(HERE / 'band_effects_from_verified_cache.csv', index=False)
    wide = results.pivot(index=['metric', 'emotion'], columns='band', values='standardized_effect')
    correlation = float(np.corrcoef(wide.broadband_1_45, wide.broadband_4_45)[0, 1])
    report = dict(cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                  table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                  n_effects=len(results), maximum_effect_difference=float(results.difference.abs().max()),
                  broadband_vs_delta_excluded_correlation=correlation,
                  matches_reported_0_62=round(correlation, 2) == .62,
                  scope='All 336 standardized effects recomputed from cache. Cache raw wPLI was sampled for 63 trials in seven bands, not every narrow-band trial; omnibus p values were not independently refit here.')
    (HERE / 'BAND_EFFECT_VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
