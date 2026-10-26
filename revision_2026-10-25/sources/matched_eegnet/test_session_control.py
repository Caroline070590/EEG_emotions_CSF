"""Train-fold session-control decisions, provenance and comparison alignment."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import report
import session_control
from common import digest, fingerprint, PRIMARY_SEED


def fixture(root):
    prepared = root / "prepared"
    directory = prepared / "seed7" / "30s"
    directory.mkdir(parents=True)
    out = root / "results"; out.mkdir()
    n_subjects, n_videos = 10, 4
    n = n_subjects * n_videos
    meta = {"trial_idx": np.arange(n), "subject": np.repeat(np.arange(1, n_subjects + 1), n_videos),
            "video": np.tile(np.arange(n_videos), n_subjects),
            "y": np.tile([0, 0, 1, 1], n_subjects), "session": np.tile([1, 1, 2, 2], n_subjects),
            "window_idx": np.arange(n), "window_id": np.zeros(n, dtype=int), "window_start": np.zeros(n, dtype=int)}
    np.savez_compressed(directory / "meta.npz", **meta)
    config = {"primary_seed": PRIMARY_SEED, "n_folds": 5,
              "datasets": {"seed7": {"n_subjects": n_subjects, "n_videos": n_videos, "class_names": ["a", "b"]}},
              "source_sha256": {"session_control.py": digest(Path(session_control.__file__))},
              "prepared_manifest": {"seed7": {"30": {"output_sha256": {"meta.npz": digest(directory / "meta.npz")}}}}}
    (out / "config.json").write_text(json.dumps(config))
    return prepared, out, config


class SessionControlTests(unittest.TestCase):
    def test_class_balanced_decision_and_no_test_label_use(self):
        sessions = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        train, test = np.arange(8), np.arange(8, 10)
        pred = session_control.fit_predict_session(sessions, y, train, test, 2)
        np.testing.assert_array_equal(pred, [1, 0])
        y[test] = 1
        np.testing.assert_array_equal(session_control.fit_predict_session(sessions, y, train, test, 2), pred)

    def test_output_provenance_and_paired_report_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared, out, config = fixture(Path(temp))
            with contextlib.redirect_stdout(io.StringIO()):
                record = session_control.build_control(prepared, out)
            self.assertEqual(record["balanced_accuracy"], 1.0)
            trial, summary, roles, files = report.load_session_control(out, config, fingerprint(config), 30)
            self.assertEqual(summary["balanced_accuracy_percent"], 100)
            self.assertEqual(len(roles), 5)
            self.assertEqual(len(files), 2)
            self.assertEqual(len(trial["y"]), 40)
            difference, values = report.paired_difference(trial, trial, 2, 30, PRIMARY_SEED)
            self.assertEqual(difference["gain_pp"], 0)
            self.assertTrue(np.all(values == 0))
            for role in roles:
                self.assertFalse(set(role["outer_train_subjects"]) & set(role["test_subjects"]))

    def test_changed_metadata_and_changed_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared, out, config = fixture(Path(temp))
            with contextlib.redirect_stdout(io.StringIO()):
                session_control.build_control(prepared, out)
            path = out / "session_control.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                report.load_session_control(out, config, fingerprint(config), 30)
            meta = prepared / "seed7" / "30s" / "meta.npz"
            meta.write_bytes(meta.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "metadata"):
                session_control.build_control(prepared, out)


if __name__ == "__main__":
    unittest.main()
