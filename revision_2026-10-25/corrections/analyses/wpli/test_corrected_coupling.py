"""Regression check against all recovered numerical Figure 2 coefficients."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'corrected_pipeline/src'))
from part2_synchrony.run_part2 import _pooled_beta
from lib.stats import emotion_slope_clustered_ols


def test_all_published_coupling_coefficients():
    d = pd.read_csv(HERE / 'historical_global_coupling_trial_inputs.csv')
    assert len(d) == 3444
    assert not d.duplicated(['subject', 'emotion', 'trial']).any()
    pooled = pd.read_csv(HERE / 'historical_global_coupling_pooled_refit.csv')
    slopes = pd.read_csv(HERE / 'historical_global_coupling_emotion_slopes_refit.csv')
    interactions = pd.read_csv(HERE / 'historical_global_coupling_interactions_refit.csv')
    for row in pooled.itertuples():
        y, x = row.topology + '_z', row.synchrony + '_z'
        np.testing.assert_allclose(_pooled_beta(d, y, x), [row.beta, row.p_nominal], atol=1e-12, rtol=1e-10)
        estimated, pi = emotion_slope_clustered_ols(d, y, x)
        subset = slopes[(slopes.topology == row.topology) & (slopes.synchrony == row.synchrony)]
        assert len(subset) == 9
        for e in subset.itertuples():
            np.testing.assert_allclose(estimated[e.emotion], [e.beta, e.p_nominal], atol=1e-12, rtol=1e-10)
        expected = interactions[(interactions.topology == row.topology) & (interactions.synchrony == row.synchrony)]
        np.testing.assert_allclose(pi, expected.interaction_p_nominal.item(), atol=1e-12, rtol=1e-10)
