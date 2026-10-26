"""Focused checks for trial aggregation, participant splits and pending inference."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import report


def window_data(subjects, seconds=10, model="eegnet", fold=0):
    records = {k: [] for k in ("y", "trial_idx", "subject", "video", "fold", "window_id", "window_start", "window_idx", "probabilities")}
    for subject in subjects:
        for video in range(2):
            trial = 2 * subject + video
            prediction = video if model in ("eegnet_connectivity", "eegnet_recurrence", "eegnet_combined") or subject % 2 == 0 else 1 - video
            for window in range(30 // seconds):
                row = {"y": video, "trial_idx": trial, "subject": subject, "video": video, "fold": fold,
                       "window_id": window, "window_start": seconds * window, "window_idx": trial * (30 // seconds) + window,
                       "probabilities": [0.9, 0.1] if prediction == 0 else [0.1, 0.9]}
                for key in records:
                    records[key].append(row[key])
    return {key: np.asarray(value) for key, value in records.items()}


def synthetic_results(root, omit=None):
    dataset = {"n_subjects": 4, "n_videos": 2, "class_names": ["a", "b"]}
    config = {"primary_seed": report.PRIMARY_SEED, "seeds": [report.PRIMARY_SEED], "n_folds": 2,
              "datasets": {"faced": dataset, "seed7": dataset}, "seconds": [10, 30], "models": list(report.MODELS)}
    (root / "config.json").write_text(json.dumps(config))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (root / "runs").mkdir(); (root / "predictions").mkdir()
    for name in config["datasets"]:
        for seconds in config["seconds"]:
            for model in config["models"]:
                for fold in range(2):
                    if omit == (name, seconds, model, fold):
                        continue
                    test = [0, 1] if fold == 0 else [2, 3]
                    outer = sorted(set(range(4)) - set(test))
                    stem = f"{name}_{seconds}s_{model}_seed{report.PRIMARY_SEED}_fold{fold}"
                    relative = f"predictions/{stem}.npz"
                    np.savez(root / relative, **window_data(test, seconds, model, fold))
                    run = {"dataset": name, "seconds": seconds, "model": model, "seed": report.PRIMARY_SEED, "fold": fold,
                           "config_sha256": config_hash, "complete": True, "predictions_path": relative,
                           "artifact_sha256": {relative: report.sha256(root / relative)},
                           "outer_train_subjects": outer, "inner_train_subjects": outer[:1], "validation_subjects": outer[1:],
                           "test_subjects": test, "selected_epoch": 2, "validation_criterion": report.VALIDATION_CRITERION,
                           "n_classes": 2, "refit_outer_train": True}
                    (root / "runs" / f"{stem}.json").write_text(json.dumps(run))
    return config, config_hash


class ReportTests(unittest.TestCase):
    def test_probability_average_not_majority_vote(self):
        data = window_data([0])
        data["probabilities"][:3] = [[.51, .49], [.51, .49], [.01, .99]]
        trial, _ = report.aggregate_windows(data, 10, 2)
        self.assertEqual(trial["pred"][0], 1)
        np.testing.assert_allclose(trial["probabilities"][0], [1.03 / 3, 1.97 / 3])

    def test_invalid_window_layout_and_probabilities_rejected(self):
        data = window_data([0])
        data["window_id"][1] = 0
        with self.assertRaisesRegex(ValueError, "windows"):
            report.aggregate_windows(data, 10, 2)
        data = window_data([0]); data["probabilities"][0] = [0.8, 0.8]
        with self.assertRaisesRegex(ValueError, "sum to one"):
            report.aggregate_windows(data, 10, 2)

    def test_sign_flip_is_two_sided_and_zero_difference_is_one(self):
        a = report.paired_sign_flip([.1, .2, .3, .4], 99, 17)
        b = report.paired_sign_flip([-.1, -.2, -.3, -.4], 99, 17)
        self.assertEqual(a, b)
        self.assertEqual(report.paired_sign_flip([0, 0, 0], 99, 17), 1)
        self.assertGreaterEqual(a, 1 / 100)
        np.testing.assert_allclose(report.holm([.01, .04, .03]), [.03, .06, .06])

    def test_full_report_and_fixed_holm_family(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); synthetic_results(root)
            with contextlib.redirect_stdout(io.StringIO()):
                result = report.build_report(root, n_bootstrap=30, n_randomizations=99, figures=False)
            self.assertEqual(result["completed_fold_runs"], 40)
            self.assertEqual(result["completed_model_runs"], 20)
            self.assertEqual(result["holm_status"], "complete")
            self.assertEqual(len(result["comparisons"]), 16)
            self.assertTrue(all(row["p_holm"] is not None for row in result["comparisons"]))
            self.assertTrue(all(row["n_trials"] == 8 for row in result["performance"]))
            self.assertTrue(all(abs(row["gain_pp"]) < 1e-10 for row in result["duration_comparisons"]))

    def test_missing_fold_withholds_scores_and_family_adjustment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); synthetic_results(root, omit=("seed7", 10, "eegnet_combined", 1))
            with contextlib.redirect_stdout(io.StringIO()):
                result = report.build_report(root, n_bootstrap=30, n_randomizations=99, figures=False)
            self.assertEqual(result["holm_status"], "pending")
            pending = [r for r in result["performance"] if r["status"] == "pending"]
            self.assertEqual(len(pending), 1)
            self.assertIsNone(pending[0]["balanced_accuracy_percent"])
            self.assertTrue(all(row["p_holm"] is None for row in result["comparisons"]))
            self.assertTrue(all(row["p_paired_sign_flip"] is None for row in result["comparisons"] if row["status"] == "pending"))

    def test_participant_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); config, config_hash = synthetic_results(root)
            run = json.loads(next((root / "runs").glob("faced_10s_eegnet_seed*_fold0.json")).read_text())
            run["validation_subjects"] = run["inner_train_subjects"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                report.validate_fold(run, window_data([0, 1]), config["datasets"]["faced"], 10, 0, config_hash)

    def test_changed_prediction_artifact_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); synthetic_results(root)
            path = next((root / "predictions").glob("*.npz"))
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                report.build_report(root, n_bootstrap=30, n_randomizations=99, figures=False)


if __name__ == "__main__":
    unittest.main()
