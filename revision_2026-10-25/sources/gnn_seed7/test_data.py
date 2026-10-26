"""SEED-VII identity, participant leakage and source-block null checks."""
import os
from pathlib import Path
import unittest

import numpy as np

from data import (CHANNELS, EMOTIONS, NODE_NAMES, VIEW_NAMES, IU,
                  edges_to_adjacency, fit_fold_scaler, load_dataset,
                  make_participant_folds, model_data, original_video_to_class,
                  permuted_video_labels, ranking_groups, video_block_permutation)


class SeedDataTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(91)
        self.subject = np.repeat(np.arange(1, 21), 80)
        self.video = np.tile(np.arange(80), 20)
        # Variable source-block sizes deliberately let clip-class counts change.
        block_labels = np.repeat(np.arange(7), [10, 9, 7, 9, 4, 11, 6])
        block_sizes = np.r_[np.full(24, 2), np.ones(32)].astype(int)
        video_to_block = np.repeat(np.arange(56), block_sizes)
        video_to_class = block_labels[video_to_block]
        self.X = rng.normal(size=(1600, 30, 10)).astype(np.float32)
        self.E = rng.normal(size=(1600, 12, 435)).astype(np.float32)
        self.data = dict(X=self.X, E=self.E, subject=self.subject, video=self.video,
                         y=video_to_class[self.video], node_names=NODE_NAMES,
                         view_names=VIEW_NAMES, channels=CHANNELS,
                         block_labels=block_labels, video_to_block=video_to_block,
                         video_to_class=video_to_class)

    def test_five_complete_participant_folds(self):
        folds = make_participant_folds(self.data)
        repeated = make_participant_folds(self.data)
        coverage = np.zeros(1600, dtype=int)
        for (train, test), (train_again, test_again) in zip(folds, repeated):
            self.assertEqual(len(np.unique(self.subject[train])), 16)
            self.assertEqual(len(np.unique(self.subject[test])), 4)
            self.assertEqual(len(np.intersect1d(self.subject[train], self.subject[test])), 0)
            np.testing.assert_array_equal(train, train_again)
            np.testing.assert_array_equal(test, test_again)
            np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(1600))
            coverage[test] += 1
        np.testing.assert_array_equal(coverage, 1)

    def test_scaling_only_uses_training_participants(self):
        train, test = make_participant_folds(self.data)[0]
        self.X[train, 0, 0] = 4.0
        self.E[train, 0, 0] = 8.0
        scaler = fit_fold_scaler(self.X, self.E, train)
        changed_x, changed_e = self.X.copy(), self.E.copy()
        changed_x[test] += 1000
        changed_e[test] -= 1000
        other = fit_fold_scaler(changed_x, changed_e, train)
        self.assertEqual(scaler.to_dict(), other.to_dict())
        x, e = scaler.transform(changed_x, changed_e)
        np.testing.assert_array_equal(x[:, 0, 0], 0)
        np.testing.assert_array_equal(e[:, 0, 0], 0)
        self.assertAlmostEqual(float(x[train, 1, 1].mean()), 0, places=6)
        self.assertAlmostEqual(float(x[train, 1, 1].std()), 1, places=6)
        self.assertGreater(float(x[test, 1, 1].mean()), 100)

    def test_adjacency_order_and_symmetry(self):
        A = edges_to_adjacency(self.E[:2])
        np.testing.assert_array_equal(A, A.swapaxes(-1, -2))
        np.testing.assert_array_equal(np.diagonal(A, axis1=-1, axis2=-2), 0)
        np.testing.assert_array_equal(A[:, :, IU[0], IU[1]], self.E[:2])
        train, _ = make_participant_folds(self.data)[0]
        result = model_data(self.data, fit_fold_scaler(self.X, self.E, train))
        self.assertEqual(result["A"].shape, (1600, 12, 30, 30))
        np.testing.assert_array_equal(result["y"], self.data["y"])

    def test_source_block_null_preserves_block_labels_not_clip_counts(self):
        y, mapping = permuted_video_labels(self.data, seed=81)
        first_clip = np.array([np.flatnonzero(self.data["video_to_block"] == b)[0] for b in range(56)])
        shuffled_block_labels = mapping[first_clip]
        np.testing.assert_array_equal(np.bincount(shuffled_block_labels, minlength=7),
                                      np.bincount(self.data["block_labels"], minlength=7))
        for b in range(56):
            self.assertEqual(len(np.unique(mapping[self.data["video_to_block"] == b])), 1)
        for v in range(80):
            np.testing.assert_array_equal(y[self.video == v], mapping[v])
        self.assertFalse(np.array_equal(np.bincount(mapping, minlength=7),
                                       np.bincount(self.data["video_to_class"], minlength=7)))
        np.testing.assert_array_equal(y, permuted_video_labels(self.data, seed=81)[0])
        np.testing.assert_array_equal(original_video_to_class(self.data), self.data["video_to_class"])

    def test_feature_donors_preserve_subject_and_shared_clip_mapping(self):
        _, test = make_participant_folds(self.data)[0]
        test = test[::-1]
        donors = video_block_permutation(self.subject, self.video, test, seed=53)
        np.testing.assert_array_equal(self.subject[donors], self.subject[test])
        np.testing.assert_array_equal(np.sort(donors), np.sort(test))
        mappings = []
        for participant in np.unique(self.subject[test]):
            mask = self.subject[test] == participant
            mappings.append(dict(zip(self.video[test[mask]], self.video[donors[mask]])))
        self.assertTrue(all(m == mappings[0] for m in mappings))
        with self.assertRaises(ValueError):
            video_block_permutation(self.subject, self.video, test[:-1], seed=53)

    def test_ranking_groups_and_nonfinite_guard(self):
        groups = ranking_groups(self.data)
        self.assertEqual(len(groups), 26)
        self.assertEqual(len({g["name"] for g in groups}), 26)
        atomic = [g for g in groups if g["kind"] == "atomic"]
        self.assertEqual(sorted(i for g in atomic for i in g["node_indices"]), list(range(10)))
        self.assertEqual(sorted(i for g in atomic for i in g["edge_indices"]), list(range(12)))
        train, _ = make_participant_folds(self.data)[0]
        self.X[train[0], 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_fold_scaler(self.X, self.E, train)

    def test_available_real_cache_identity_and_schema(self):
        cache = Path(os.environ.get("SEED7_CACHE", str(Path(__file__).resolve().parent.parent / "seed7_replication/cache")))
        if not (cache / "features_arrays.npz").exists():
            self.skipTest("Set SEED7_CACHE to run real-cache provenance validation")
        d = load_dataset(cache)
        self.assertEqual(d["X"].shape, (1600, 30, 10))
        self.assertEqual(d["E"].shape, (1600, 12, 435))
        self.assertEqual(d["manifest"]["n_classes"], 7)
        self.assertEqual(d["manifest"]["emotions"], EMOTIONS)
        self.assertEqual(len(d["manifest"]["video_to_session"]), 80)
        self.assertEqual(d["manifest"]["n_source_condition_blocks"], 56)
        np.testing.assert_array_equal(d["y"], d["video_to_class"][d["video"]])
        np.testing.assert_array_equal(d["session"], d["video_to_session"][d["video"]])
        self.assertEqual(set(d["manifest"]["mixed_source_programmes"]),
                         {"under the hawthorn tree", "you are my life"})
        for seed in (1, 11, 99):
            y, mapping = permuted_video_labels(d, seed)
            self.assertEqual(len(np.unique(y)), 7)
            self.assertEqual(mapping.shape, (80,))


if __name__ == "__main__":
    unittest.main()
