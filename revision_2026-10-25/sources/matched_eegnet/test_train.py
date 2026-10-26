"""Leakage, pairing, reproducibility, and actual CPU software-smoke checks."""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import train
from common import DATASETS, MODELS, PRIMARY_SEED, SEEDS, digest, fingerprint, participant_split, side_donors


def metadata(n_subjects=10, n_classes=2, windows=3):
    subject = np.repeat(np.arange(n_subjects), n_classes * windows)
    video = np.tile(np.repeat(np.arange(n_classes), windows), n_subjects)
    n = len(subject)
    return {"subject": subject, "video": video, "y": video.copy(),
            "trial_idx": np.repeat(np.arange(n_subjects * n_classes), windows),
            "window_idx": np.arange(n), "window_id": np.tile(np.arange(windows), n // windows),
            "window_start": np.tile(np.arange(windows) * (30 // windows), n // windows)}


def fixture(directory, model="eegnet_shuffled"):
    prepared = directory / "prepared" / "faced" / "10s"
    prepared.mkdir(parents=True)
    meta = metadata()
    rng = np.random.default_rng(91)
    signal = rng.normal(size=(len(meta["y"]), 30, 2000)).astype(np.float32)
    side = rng.normal(size=(len(meta["y"]), 5370)).astype(np.float32)
    np.save(prepared / "signal.npy", signal)
    np.save(prepared / "side.npy", side)
    np.savez(prepared / "meta.npz", **meta)
    manifest = {"dataset": "faced", "seconds": 10,
                "output_sha256": {name: digest(prepared / name) for name in ("signal.npy", "side.npy", "meta.npz")}}
    (prepared / "MANIFEST.json").write_text(json.dumps(manifest))
    pinned = dict(manifest, manifest_sha256=digest(prepared / "MANIFEST.json"))
    config = {"primary_seed": PRIMARY_SEED, "n_folds": 5, "models": list(MODELS), "seeds": list(SEEDS),
              "seconds": [10, 30], "datasets": {"faced": {"n_subjects": 10, "n_videos": 2, "class_names": ["a", "b"]}},
              "tasks": [{"dataset": "faced", "seconds": 10, "model": model, "seed": PRIMARY_SEED, "fold": 0}],
              "source_sha256": {name: digest(train.HERE / name) for name in train.REQUIRED_SOURCES},
              "training": dict(train.TRAINING_DEFAULTS), "prepared_manifest": {"faced": {"10": pinned}}}
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config))
    args = argparse.Namespace(prepared=directory / "prepared", config=config_path,
                              out=directory / "out", task=0, device="cpu", smoke=True,
                              smoke_epochs=1, threads=1)
    return args, config, signal, side, meta


class TrainingTests(unittest.TestCase):
    def test_train_only_scalers_population_moments_and_constant_features(self):
        rng = np.random.default_rng(12)
        signal = rng.normal(size=(8, 30, 20)).astype(np.float32)
        side = rng.normal(size=(8, 5370)).astype(np.float32)
        rows = np.array([0, 2, 4, 6])
        signal[rows, 0, :] = 5
        side[rows, 3] = 7
        fit = train.fit_scalers(signal, side, rows, chunk_size=2)
        np.testing.assert_allclose(fit["raw_mean"], signal[rows].astype(float).mean(axis=(0, 2)), atol=1e-7)
        np.testing.assert_allclose(fit["raw_std"][1:], signal[rows, 1:].astype(float).std(axis=(0, 2)), rtol=1e-6)
        np.testing.assert_allclose(fit["side_mean"], side[rows].astype(float).mean(axis=0), atol=1e-7)
        untouched = np.setdiff1d(np.arange(8), rows)
        signal[untouched] = 99999; side[untouched] = -99999
        changed = train.fit_scalers(signal, side, rows, chunk_size=3)
        for key in fit:
            np.testing.assert_allclose(changed[key], fit[key], rtol=1e-6, atol=1e-7)
        meta = {"y": np.zeros(8, dtype=int)}
        raw, features, _ = train.make_batch(signal, side, meta, untouched, fit, "eegnet_combined", "cpu")
        self.assertTrue(torch.all(raw[:, 0] == 0))
        self.assertTrue(torch.all(features[:, 3] == 0))
        self.assertTrue(torch.all(raw[:, 1] > 1000))  # no clipping

    def test_exact_feature_slices_class_weights_and_whole_trial_donors(self):
        meta = metadata()
        n = len(meta["y"])
        raw = np.ones((n, 30, 2), dtype=np.float32)
        side = np.arange(n * 5370, dtype=np.float32).reshape(n, 5370)
        scalers = {"raw_mean": np.zeros(30), "raw_std": np.ones(30), "raw_constant": np.zeros(30, bool),
                   "side_mean": np.zeros(5370), "side_std": np.ones(5370), "side_constant": np.zeros(5370, bool)}
        rows = np.arange(n)
        donors = side_donors(meta, rows, PRIMARY_SEED)
        np.testing.assert_array_equal(meta["subject"][donors], meta["subject"])
        np.testing.assert_array_equal(meta["window_id"][donors], meta["window_id"])
        for trial in np.unique(meta["trial_idx"]):
            self.assertEqual(len(np.unique(meta["trial_idx"][donors[meta["trial_idx"] == trial]])), 1)
        for variant, selection in train.SIDE_SLICES.items():
            _, features, _ = train.make_batch(raw, side, meta, rows, scalers, variant, "cpu", donors)
            if selection is None:
                self.assertIsNone(features)
            else:
                np.testing.assert_array_equal(features.numpy(), side[donors, selection])
        y = np.array([0, 0, 0, 1, 2, 2, 2, 2, 2])
        np.testing.assert_allclose(train.class_weights(y, np.arange(4), 2), [2 / 3, 2])
        with self.assertRaisesRegex(ValueError, "Every training"):
            train.class_weights(y, np.arange(3), 2)

    def test_trial_selection_uses_averaged_probability_and_unweighted_loss(self):
        # Majority window predictions are wrong for class 0, but their mean is correct.
        probability = np.array([[.49, .51], [.49, .51], [.99, .01],
                                [.2, .8], [.3, .7], [.4, .6]])
        meta = {"trial_idx": np.repeat([10, 11], 3), "y": np.repeat([0, 1], 3)}
        accuracy, loss = train.trial_metrics(probability, meta, np.arange(6))
        self.assertEqual(accuracy, 1)
        self.assertAlmostEqual(loss, -np.log([(.49 + .49 + .99) / 3, .7]).mean())
        self.assertTrue(train.improves(.5, .9, .5, 1))
        self.assertFalse(train.improves(.49, .1, .5, 1))
        self.assertTrue(train.improves(.51, 100, .5, 1))
        self.assertFalse(train.improves(.5, 1, .5, 1))

    def test_smoke_splits_keep_participant_roles_and_complete_windows(self):
        meta = metadata(n_subjects=20, n_classes=7)
        split = participant_split(meta["subject"], 2, PRIMARY_SEED)
        small = train.smoke_splits(meta, split)
        subjects = {role: set(meta["subject"][rows]) for role, rows in small.items()}
        self.assertFalse(subjects["inner_train"] & subjects["validation"])
        self.assertFalse(subjects["outer_train"] & subjects["test"])
        self.assertEqual(subjects["outer_train"], subjects["inner_train"] | subjects["validation"])
        for role in ("inner_train", "validation", "test"):
            self.assertEqual(len(subjects[role]), 2)
            self.assertEqual(len(small[role]), 2 * 7 * 3)
            for trial in np.unique(meta["trial_idx"][small[role]]):
                rows = small[role][meta["trial_idx"][small[role]] == trial]
                np.testing.assert_array_equal(meta["window_id"][rows], [0, 1, 2])

    def test_source_and_metadata_provenance_and_production_cpu_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, *_ = fixture(Path(tmp))
            task, _ = train.validate_config(config, 0, smoke=True)
            train.load_prepared(args.prepared, task, config, smoke=True)
            altered = dict(config, source_sha256=dict(config["source_sha256"], **{"models.py": "bad"}))
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                train.validate_config(altered, 0, smoke=True)
            metadata_path = args.prepared / "faced" / "10s" / "meta.npz"
            with metadata_path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "metadata hash changed"):
                train.load_prepared(args.prepared, task, config, smoke=True)
            config["datasets"] = DATASETS
            config["tasks"] = [dict(zip(("dataset", "seconds", "model", "seed", "fold"), values))
                               for values in itertools.product(DATASETS, (10, 30), MODELS, SEEDS, range(5))]
            args.config.write_text(json.dumps(config)); args.smoke = False
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                train.run_task(args)

    def test_actual_forward_backward_refit_artifacts_and_smoke_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, config, signal, side, meta = fixture(Path(tmp))
            initial_states = []
            actual_builder = train.build_model

            def recording_builder(*a, **kw):
                model = actual_builder(*a, **kw)
                initial_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
                return model

            with mock.patch.object(train, "build_model", side_effect=recording_builder), contextlib.redirect_stdout(io.StringIO()):
                result = train.run_task(args)
            self.assertEqual(len(initial_states), 2)
            for name in initial_states[0]:
                torch.testing.assert_close(initial_states[0][name], initial_states[1][name], rtol=0, atol=0)
            self.assertFalse(result["complete"])
            self.assertFalse(result["production_complete"])
            self.assertTrue(result["smoke_complete"])
            self.assertEqual(result["selected_epoch"], 1)
            self.assertEqual(result["selection_epochs"], 1)
            self.assertEqual(result["refit_epochs"], 1)
            self.assertEqual(result["config_sha256"], fingerprint(config))
            self.assertFalse((args.out / "runs").exists())
            self.assertFalse((args.out / "predictions").exists())
            self.assertFalse((args.out / "config.json").exists())
            for relative, expected in result["artifact_sha256"].items():
                self.assertTrue(relative.startswith("smoke_"))
                self.assertEqual(digest(args.out / relative), expected)
            with np.load(args.out / result["predictions_path"]) as prediction:
                self.assertEqual(prediction["probabilities"].shape, (12, 2))
                np.testing.assert_allclose(prediction["probabilities"].sum(1), 1, atol=1e-6)
                self.assertEqual(set(prediction["subject"]), set(result["test_subjects"]))
            with np.load(args.out / result["donors_path"]) as donors:
                self.assertNotEqual(int(donors["validation_seed"]), int(donors["test_seed"]))
                for stage in ("selection_train", "refit_train"):
                    rows = donors[stage + "_rows"]
                    mapping = donors[stage + "_donors"][0]
                    np.testing.assert_array_equal(np.sort(mapping), rows)
                    np.testing.assert_array_equal(meta["subject"][mapping], meta["subject"][rows])
                    np.testing.assert_array_equal(meta["window_id"][mapping], meta["window_id"][rows])
            with np.load(args.out / result["splits_path"]) as splits, np.load(args.out / result["scalers_path"]) as scalers:
                for prefix, role in (("inner", "inner_train"), ("refit", "outer_train")):
                    np.testing.assert_allclose(scalers[prefix + "_side_mean"],
                                               side[splits[role]].astype(float).mean(0), atol=1e-7)
            final = torch.load(args.out / result["refit_path"], map_location="cpu", weights_only=False)
            self.assertGreater(float((final["state_dict"]["temporal_conv.weight"] - initial_states[1]["temporal_conv.weight"]).abs().sum()), 0)
            self.assertGreater(float(final["state_dict"]["side_classifier.weight"].abs().sum()), 0)
            self.assertTrue(torch.all(final["state_dict"]["raw_classifier.weight"].norm(dim=1) <= .25 + 1e-6))
            # Completed smoke reruns verify artifacts and do not train again.
            with mock.patch.object(train, "build_model", side_effect=AssertionError("should reuse")), contextlib.redirect_stdout(io.StringIO()):
                again = train.run_task(args)
            self.assertEqual(again["artifact_sha256"], result["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
