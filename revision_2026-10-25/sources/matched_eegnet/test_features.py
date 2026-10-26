"""Numerical parity and input-matching checks for the new window extractor.

The loop reference below is frozen from the earlier RQA definitions. When the
original local library is available, tests also load that actual module. This
keeps deployed tests self-contained without replacing the independent reference
by a second implementation of the new vectorized line-counting method.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch

from features import (BANDS, CONFIG, IU, TARGET_RR, THEILER, _summarize_rqa125,
                      extract_from_signal, preprocess_window, process_window,
                      recurrence_line_lengths, recurrence_matrix, rqa_metrics,
                      synchronization_edges)


def _legacy_lines(R):
    n = len(R)
    diag, vert = [], []
    for k in range(THEILER + 1, n):
        d = np.diagonal(R, k).astype(np.int8)
        if d.any():
            e = np.diff(np.r_[0, d, 0])
            diag += list(np.where(e == -1)[0] - np.where(e == 1)[0])
    for j in range(n):
        c = R[:, j].astype(np.int8)
        if c.any():
            e = np.diff(np.r_[0, c, 0])
            vert += list(np.where(e == -1)[0] - np.where(e == 1)[0])
    return np.array(diag), np.array(vert)


def _legacy_rqa(R):
    dl, vl = _legacy_lines(R)
    n_rec_upper = R[np.triu_indices(len(R), THEILER + 1)].sum()
    dl2, vl2 = dl[dl >= 2], vl[vl >= 2]
    det = dl2.sum() / n_rec_upper if n_rec_upper else np.nan
    lam = vl2.sum() / R.sum() if R.sum() else np.nan
    if len(dl2):
        _, cnt = np.unique(dl2, return_counts=True)
        p = cnt / cnt.sum()
        ent = float(-(p * np.log(p)).sum())
    else:
        ent = 0.0
    return [det, lam, dl2.mean() if len(dl2) else 0.0, ent,
            vl2.mean() if len(vl2) else 0.0]


class MatchedWindowFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(39)
        time = np.arange(7500) / 250.0
        frequencies = np.linspace(3.5, 35, 30)
        cls.raw = np.sin(2 * np.pi * frequencies[:, None] * time) + rng.normal(scale=0.4, size=(30, 7500))
        cls.original_raw = cls.raw.copy()
        cls.short = process_window(cls.raw[:, :2500])
        cls.long = process_window(cls.raw)
        cls.legacy = None
        path = Path(__file__).resolve().parent.parent / "revision_2026-10/code/10_recurrence_features.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("earlier_rqa_reference", path)
            cls.legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.legacy)

    def test_common_preprocessing_matches_explicit_signal_formula(self):
        expected = sosfiltfilt(butter(4, [1, 45], btype="bandpass", fs=200, output="sos"),
                              resample_poly(self.raw[:, :2500], 4, 5, axis=1), axis=1).astype(np.float32)
        np.testing.assert_array_equal(self.short["signal"], expected)
        np.testing.assert_array_equal(self.raw, self.original_raw)

    def test_shapes_ordering_and_all_time_coverage_for_both_durations(self):
        for result, seconds, n_blocks in [(self.short, 10, 5), (self.long, 30, 15)]:
            with self.subTest(seconds=seconds):
                for key, shape in [("signal", (30, 200 * seconds)), ("X", (30, 10)),
                                   ("E", (12, 435)), ("connectivity", (4785,)),
                                   ("recurrence", (585,)), ("logpower", (150,)),
                                   ("full_features", (5520,))]:
                    self.assertEqual(result[key].shape, shape)
                    self.assertEqual(result[key].dtype, np.float32)
                    self.assertTrue(np.isfinite(result[key]).all())
                self.assertEqual(result["metadata"]["rqa_n_subwindows"], n_blocks)
                self.assertEqual(n_blocks * CONFIG["rqa_block_seconds"], seconds)
                np.testing.assert_array_equal(result["connectivity"], result["E"][:11].ravel())
                np.testing.assert_array_equal(result["recurrence"][:150], result["X"][:, 5:].ravel())
                np.testing.assert_array_equal(result["recurrence"][150:], result["E"][11])
                np.testing.assert_array_equal(result["full_features"], np.r_[result["X"].ravel(), result["E"].ravel()])

    def test_power_uses_the_exact_returned_signal(self):
        f, p = welch(self.short["signal"].astype(np.float64), fs=200,
                     nperseg=400, noverlap=200, axis=1)
        expected = np.stack([np.log10(p[:, (f >= lo) & (f < hi)].mean(axis=1))
                             for lo, hi in BANDS.values()], axis=1).astype(np.float32)
        np.testing.assert_array_equal(self.short["X"][:, :5], expected)

    def test_sync_matches_original_complex_formula(self):
        Z = hilbert(self.short["signal"].astype(np.float64), axis=1).T
        Zc, Zd = Z[:, IU[0]], Z[:, IU[1]]
        ac, ad = np.abs(Zc), np.abs(Zd)
        imaginary = (Zc * np.conj(Zd)).imag
        den = np.abs(imaginary).mean(axis=0)
        expected_w = np.divide(np.abs(imaginary.mean(axis=0)), den,
                               out=np.zeros_like(den), where=den > 0)
        def corr(a, b):
            a = a - a.mean(axis=0)
            b = b - b.mean(axis=0)
            return (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0) + 1e-30)
        yo = np.abs((Zd * np.conj(Zc) / np.maximum(ac, 1e-20)).imag)
        xo = np.abs((Zc * np.conj(Zd) / np.maximum(ad, 1e-20)).imag)
        expected_o = 0.5 * (corr(ac, yo) + corr(ad, xo))
        for pair_batch in [1, 48, 435]:
            actual_w, actual_o = synchronization_edges(Z.T, pair_batch=pair_batch)
            np.testing.assert_allclose(actual_w, expected_w, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual_o, expected_o, rtol=1e-12, atol=1e-12)

    def test_vectorized_rqa_matches_existing_definitions(self):
        rng = np.random.default_rng(21)
        examples = [np.zeros((18, 18), dtype=bool),
                    recurrence_matrix(rng.normal(size=250)),
                    recurrence_matrix(np.sin(np.arange(250) * 0.27)),
                    recurrence_matrix(np.zeros(250))]
        for R in examples:
            new_diagonal, new_vertical = recurrence_line_lengths(R)
            old_diagonal, old_vertical = _legacy_lines(R)
            np.testing.assert_array_equal(new_diagonal, old_diagonal)
            np.testing.assert_array_equal(new_vertical, old_vertical)
            reference = np.nan_to_num(_legacy_rqa(R), nan=0.0)
            np.testing.assert_allclose(rqa_metrics(R), reference, rtol=1e-13, atol=1e-13)
            if self.legacy is not None:
                np.testing.assert_allclose(rqa_metrics(R), np.nan_to_num(self.legacy.rqa(R), nan=0.0), rtol=1e-13, atol=1e-13)
        if self.legacy is not None:
            x = rng.normal(size=250)
            np.testing.assert_array_equal(recurrence_matrix(x), self.legacy.recurrence(x))

    def test_same_two_second_rqa_estimator_for_both_durations(self):
        block = np.random.default_rng(73).normal(size=(30, 250))
        short_rqa, short_jrs = _summarize_rqa125(np.tile(block, (1, 5)))
        long_rqa, long_jrs = _summarize_rqa125(np.tile(block, (1, 15)))
        np.testing.assert_allclose(short_rqa, long_rqa, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(short_jrs, long_jrs, rtol=1e-13, atol=1e-13)
        normalized = (block - block.mean(axis=1, keepdims=True)) / (block.std(axis=1, keepdims=True) + 1e-12)
        matrices = [recurrence_matrix(x) for x in normalized]
        flat = np.stack([R.ravel() for R in matrices]).astype(np.float32)
        expected_joint = ((flat @ flat.T) / flat.shape[1] / TARGET_RR)[IU]
        np.testing.assert_allclose(short_jrs, expected_joint, rtol=1e-13, atol=1e-13)

    def test_other_windows_cannot_change_extracted_features(self):
        # Exercise a second unrelated input, then the original input again.
        process_window(self.raw[:, 2500:5000] * 3.0)
        again = process_window(self.raw[:, :2500])
        for name in ["signal", "X", "E", "connectivity", "recurrence"]:
            np.testing.assert_array_equal(again[name], self.short[name])
        # Extraction from the exact saved waveform also reproduces every branch.
        from_saved_signal = extract_from_signal(self.short["signal"])
        np.testing.assert_array_equal(from_saved_signal["full_features"], self.short["full_features"])

    def test_constant_signal_is_finite_and_invalid_inputs_rejected(self):
        result = process_window(np.zeros((30, 2500)))
        self.assertEqual(result["metadata"]["zero_variance_channels"], 30)
        for name in ["signal", "X", "E", "connectivity", "recurrence"]:
            self.assertTrue(np.isfinite(result[name]).all())
        np.testing.assert_array_equal(result["E"][:11], 0.0)
        for invalid in [np.zeros((29, 2500)), np.zeros((30, 500)), np.full((30, 2500), np.nan)]:
            with self.assertRaises(ValueError):
                preprocess_window(invalid)


if __name__ == "__main__":
    unittest.main()
