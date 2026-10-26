"""Numerical checks for block importance and linear score updates."""
import unittest
from types import SimpleNamespace
import numpy as np
from data import edges_to_adjacency, NODE_NAMES, VIEW_NAMES, ranking_groups, video_block_permutation
from run import unique_features, feature_columns, rank_fold


class ImportanceTests(unittest.TestCase):
    def test_linear_perturbation_equals_full_input_recomputation(self):
        rng = np.random.default_rng(78)
        raw = {'subject': np.repeat(np.arange(2), 28), 'video': np.tile(np.arange(28), 2),
               'node_names': NODE_NAMES, 'view_names': VIEW_NAMES}
        d = {'X': rng.normal(size=(56,30,10)).astype(np.float32),
             'A': edges_to_adjacency(rng.normal(size=(56,12,435))), 'y': np.zeros(56)}
        features = unique_features(d).astype(np.float64)
        coef = rng.normal(size=(9,5520))
        intercept = rng.normal(size=9)
        model = SimpleNamespace(coef_=coef, classes_=np.arange(9),
                                decision_function=lambda x: x @ coef.T + intercept)
        te = np.arange(56)
        groups = ranking_groups(raw)
        actual = rank_fold('lda_full', model, d, raw, te, groups, [19], None, features)
        donors = video_block_permutation(raw['subject'],raw['video'],te,19)
        for gi,group in enumerate(groups):
            changed = {'X': d['X'].copy(), 'A': d['A'].copy(), 'y': d['y']}
            for node in group['node_indices']: changed['X'][:,:,node]=d['X'][donors,:,node]
            for edge in group['edge_indices']: changed['A'][:,edge]=d['A'][donors,edge]
            expected = model.decision_function(unique_features(changed)).argmax(1)
            np.testing.assert_array_equal(actual[gi,0],expected)
            cols=feature_columns(group)
            self.assertEqual(len(cols),len(set(cols.tolist())))


if __name__ == '__main__':
    unittest.main()
