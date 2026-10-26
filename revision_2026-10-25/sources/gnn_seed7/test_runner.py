"""SEED-VII block-importance and linear-update checks on two synthetic participants.

Adapted from participant_prediction_2026-10-24/test_runner.py for seven classes,
80 videos and the unchanged 5,520-input representation. No EEG data or training
runs are required. Neural checks exercise the actual frozen architecture branches.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data import edges_to_adjacency, NODE_NAMES, VIEW_NAMES, ranking_groups, video_block_permutation
import run as runner
from run import unique_features, feature_columns, rank_fold
from sdgnn import HybridGNN, SummaryFirstGNN, IU


class Seed7ImportanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        rng = np.random.default_rng(78)
        mapping = np.repeat(np.arange(7), [12, 12, 12, 12, 12, 12, 8])
        self.raw = {
            'subject': np.repeat(np.arange(2), 80),
            'video': np.tile(np.arange(80), 2),
            'node_names': NODE_NAMES, 'view_names': VIEW_NAMES,
        }
        self.d = {
            'X': rng.normal(size=(160, 30, 10)).astype(np.float32),
            'A': edges_to_adjacency(rng.normal(size=(160, 12, 435))),
            'y': mapping[self.raw['video']],
        }
        self.features = unique_features(self.d).astype(np.float64)
        self.te = rng.permutation(160)  # Global donors must align with non-sorted test rows.
        self.groups = ranking_groups(self.raw)
        self.seeds = [19, 41]

    def changed_inputs(self, group, seed):
        donors = video_block_permutation(self.raw['subject'], self.raw['video'], self.te, seed)
        changed = {key: self.d[key][self.te].copy() for key in ('X', 'A', 'y')}
        for node in group['node_indices']:
            changed['X'][:, :, node] = self.d['X'][donors, :, node]
        for edge in group['edge_indices']:
            changed['A'][:, edge] = self.d['A'][donors, edge]
        return changed, donors

    def test_seven_class_linear_updates_equal_full_recomputation_all_groups(self):
        rng = np.random.default_rng(182)
        coef, intercept = rng.normal(size=(7, 5520)), rng.normal(size=7)
        model = SimpleNamespace(coef_=coef, classes_=np.arange(7),
                                decision_function=lambda x: x @ coef.T + intercept)
        actual = rank_fold('lda_full', model, self.d, self.raw, self.te, self.groups,
                           self.seeds, None, self.features)
        self.assertEqual(actual.shape, (26, 2, 160))
        self.assertEqual(actual.dtype, np.uint8)
        base_scores = model.decision_function(self.features[self.te])
        for rep, seed in enumerate(self.seeds):
            for gi, group in enumerate(self.groups):
                with self.subTest(group=group['name'], repeat=rep):
                    changed, donors = self.changed_inputs(group, seed)
                    expected_scores = model.decision_function(unique_features(changed))
                    columns = feature_columns(group)
                    delta = self.features[np.ix_(donors, columns)] - self.features[np.ix_(self.te, columns)]
                    updated_scores = base_scores + delta @ coef[:, columns].T
                    np.testing.assert_allclose(updated_scores, expected_scores, atol=2e-11, rtol=2e-12)
                    np.testing.assert_array_equal(actual[gi, rep], expected_scores.argmax(axis=1))
                    self.assertEqual(len(columns), len(set(columns.tolist())))
                    self.assertTrue(np.all((columns >= 0) & (columns < 5520)))
                    np.testing.assert_array_equal(self.raw['subject'][donors], self.raw['subject'][self.te])
                    untouched = np.setdiff1d(np.arange(5520), columns)
                    np.testing.assert_array_equal(unique_features(changed)[:, untouched],
                                                  self.features[np.ix_(self.te, untouched)])

    def test_neural_permutations_reach_graph_and_summary_and_restore_inputs(self):
        # Atomic node/edge examples plus all six nonoverlapping input blocks,
        # including recurrence nodes and the joint-recurrence edge view.
        selected = [self.groups[i] for i in (0, 10, 20, 21, 22, 23, 24, 25)]
        original_probability = runner.neural_probability
        original_x, original_a = self.d['X'].copy(), self.d['A'].copy()
        for name, constructor in (('hybrid_gnn', HybridGNN), ('sfgnn', SummaryFirstGNN)):
            with self.subTest(model=name):
                torch.manual_seed(27)
                model = constructor(n_views=12, n_feat=10, n_cls=7, n_videos=80)
                model.eval()
                original_encode = model.encode
                summary_module = model.summ if name == 'hybrid_gnn' else model.norm
                state = {'calls': 0, 'graph_calls': 0, 'summary_calls': 0, 'expected': None}
                expected_predictions = np.empty((len(selected), len(self.seeds), len(self.te)), dtype=np.uint8)

                def inspect_graph(X, A, view_mask=None):
                    expected = state['expected']
                    np.testing.assert_array_equal(X.detach().numpy(), expected['X'])
                    np.testing.assert_array_equal(A.detach().numpy(), expected['A'])
                    state['graph_calls'] += 1
                    return original_encode(X, A, view_mask)

                def inspect_summary(_module, inputs):
                    expected = state['expected']
                    X, A = torch.from_numpy(expected['X']), torch.from_numpy(expected['A'])
                    expected_summary = torch.cat([A[:, :, IU[0], IU[1]].mean(-1), X.mean(1)], 1)
                    torch.testing.assert_close(inputs[0], expected_summary, rtol=0, atol=0)
                    state['summary_calls'] += 1

                def inspect_probability(fitted, X, A, batch=256):
                    rep, gi = divmod(state['calls'], len(selected))
                    expected, donors = self.changed_inputs(selected[gi], self.seeds[rep])
                    state['expected'] = expected
                    np.testing.assert_array_equal(X, expected['X'])
                    np.testing.assert_array_equal(A, expected['A'])
                    np.testing.assert_array_equal(self.raw['subject'][donors], self.raw['subject'][self.te])
                    probability = original_probability(fitted, X, A, batch=batch)
                    self.assertEqual(probability.shape, (160, 7))
                    self.assertTrue(np.isfinite(probability).all())
                    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)
                    expected_predictions[gi, rep] = probability.argmax(axis=1)
                    state['calls'] += 1
                    return probability

                hook = summary_module.register_forward_pre_hook(inspect_summary)
                try:
                    with patch.object(model, 'encode', side_effect=inspect_graph), \
                            patch.object(runner, 'neural_probability', side_effect=inspect_probability):
                        actual = rank_fold(name, model, self.d, self.raw, self.te, selected,
                                           self.seeds, baseline_prob=None)
                finally:
                    hook.remove()
                np.testing.assert_array_equal(actual, expected_predictions)
                self.assertEqual(state['calls'], len(selected) * len(self.seeds))
                self.assertEqual(state['graph_calls'], state['calls'])
                self.assertEqual(state['summary_calls'], state['calls'])
                np.testing.assert_array_equal(self.d['X'], original_x)
                np.testing.assert_array_equal(self.d['A'], original_a)

    def test_donor_video_mapping_is_shared_for_both_participants_and_repeats(self):
        mappings = []
        for seed in self.seeds:
            donors = video_block_permutation(self.raw['subject'], self.raw['video'], self.te, seed)
            again = video_block_permutation(self.raw['subject'], self.raw['video'], self.te, seed)
            np.testing.assert_array_equal(donors, again)
            np.testing.assert_array_equal(np.sort(donors), np.sort(self.te))
            by_participant = []
            for subject in (0, 1):
                take = self.raw['subject'][self.te] == subject
                pairs = sorted(zip(self.raw['video'][self.te[take]], self.raw['video'][donors[take]]))
                by_participant.append(pairs)
            self.assertEqual(by_participant[0], by_participant[1])
            self.assertEqual(len(by_participant[0]), 80)
            self.assertTrue(any(original != donor for original, donor in by_participant[0]))
            mappings.append(by_participant[0])
        self.assertNotEqual(mappings[0], mappings[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
