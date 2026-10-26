"""Focused leakage, grouping, and adjacency checks for participant holdout."""
import unittest

import numpy as np

from data import (CHANNELS, NODE_NAMES, VIEW_NAMES, IU, edges_to_adjacency,
                  fit_fold_scaler, make_participant_folds, model_data,
                  original_video_to_class, permuted_video_labels, ranking_groups,
                  video_block_permutation)


class ParticipantDataTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.subject = np.repeat(np.arange(11), 28)
        self.video = np.tile(np.arange(28), 11)
        self.X = rng.normal(size=(308, 30, 10)).astype(np.float32)
        self.E = rng.normal(size=(308, 12, 435)).astype(np.float32)
        self.data = dict(X=self.X, E=self.E, subject=self.subject, video=self.video,
                         y=original_video_to_class()[self.video],
                         node_names=NODE_NAMES, view_names=VIEW_NAMES, channels=CHANNELS)

    def test_complete_disjoint_folds_reproducible(self):
        folds = make_participant_folds(self.data)
        repeated = make_participant_folds(self.data)
        coverage = np.zeros(len(self.subject), int)
        for (train, test), (again_train, again_test) in zip(folds, repeated):
            self.assertEqual(len(np.intersect1d(self.subject[train], self.subject[test])), 0)
            np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(308))
            np.testing.assert_array_equal(train, again_train)
            np.testing.assert_array_equal(test, again_test)
            coverage[test] += 1
        np.testing.assert_array_equal(coverage, 1)

    def test_scaler_ignores_heldout_and_zeros_training_constants(self):
        train, test = make_participant_folds(self.data)[0]
        self.X[train, 0, 0] = 4.0
        self.E[train, 0, 0] = 8.0
        scaler = fit_fold_scaler(self.X, self.E, train)
        changed_x, changed_e = self.X.copy(), self.E.copy()
        changed_x[test] += 10000
        changed_e[test] -= 10000
        changed_scaler = fit_fold_scaler(changed_x, changed_e, train)
        self.assertEqual(scaler.to_dict(), changed_scaler.to_dict())
        scaled_x, scaled_e = scaler.transform(changed_x, changed_e)
        np.testing.assert_array_equal(scaled_x[:, 0, 0], 0)
        np.testing.assert_array_equal(scaled_e[:, 0, 0], 0)
        self.assertAlmostEqual(float(scaled_x[train, 1, 1].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(scaled_x[train, 1, 1].std()), 1.0, places=6)
        self.assertGreater(float(scaled_x[test, 1, 1].mean()), 100)

    def test_adjacencies_are_symmetric_zero_diagonal_and_exact(self):
        A = edges_to_adjacency(self.E[:2])
        self.assertEqual(A.shape, (2, 12, 30, 30))
        np.testing.assert_array_equal(A, A.swapaxes(-1, -2))
        np.testing.assert_array_equal(np.diagonal(A, axis1=-2, axis2=-1), 0)
        np.testing.assert_array_equal(A[:, :, IU[0], IU[1]], self.E[:2])
        train, _ = make_participant_folds(self.data)[0]
        model = model_data(self.data, fit_fold_scaler(self.X, self.E, train))
        self.assertEqual(model["A"].shape, (308, 12, 30, 30))
        np.testing.assert_array_equal(model["y"], self.data["y"])

    def test_shared_video_block_shuffle_stays_within_participant(self):
        _, test = make_participant_folds(self.data)[0]
        test = test[::-1]  # The helper must not rely on sorted rows.
        donors = video_block_permutation(self.subject, self.video, test, seed=53)
        np.testing.assert_array_equal(self.subject[donors], self.subject[test])
        np.testing.assert_array_equal(np.sort(donors), np.sort(test))
        np.testing.assert_array_equal(donors, video_block_permutation(self.subject, self.video, test, seed=53))
        self.assertTrue(np.any(self.video[donors] != self.video[test]))
        mappings = []
        for participant in np.unique(self.subject[test]):
            mask = self.subject[test] == participant
            mappings.append(dict(zip(self.video[test[mask]], self.video[donors[mask]])))
        self.assertTrue(all(mapping == mappings[0] for mapping in mappings))
        with self.assertRaises(ValueError):
            video_block_permutation(self.subject, self.video, test[:-1], seed=53)

    def test_null_preserves_video_counts_and_shared_video_labels(self):
        y, mapping = permuted_video_labels(self.video, seed=81)
        np.testing.assert_array_equal(np.bincount(mapping), np.bincount(original_video_to_class()))
        np.testing.assert_array_equal(np.bincount(y), np.bincount(self.data["y"]))
        for video in range(28):
            np.testing.assert_array_equal(y[self.video == video], mapping[video])
        self.assertFalse(np.array_equal(mapping, original_video_to_class()))

    def test_ranking_groups_cover_each_atomic_input_once(self):
        groups = ranking_groups(self.data)
        self.assertEqual(len(groups), 26)
        self.assertEqual(len({g["name"] for g in groups}), 26)
        atomic = [g for g in groups if g["kind"] == "atomic"]
        self.assertEqual(len(atomic), 22)
        self.assertEqual(sorted(i for g in atomic for i in g["node_indices"]), list(range(10)))
        self.assertEqual(sorted(i for g in atomic for i in g["edge_indices"]), list(range(12)))

    def test_nonfinite_training_data_rejected(self):
        train, _ = make_participant_folds(self.data)[0]
        self.X[train[0], 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_fold_scaler(self.X, self.E, train)


if __name__ == "__main__":
    unittest.main()
