"""Matched-window EEG inputs and fixed recurrence/connectivity descriptors.

One 30-channel, 10- or 30-second input at 250 Hz is independently resampled
4/5 to 200 Hz and filtered with a fourth-order 1--45 Hz zero-phase Butterworth
filter. The returned float32 signal is the common input to EEGNet and all
descriptors; descriptor calculations use its values in float64 arithmetic.

RQA uses a NEW fixed two-second subwindow setting for both input durations:
resample this signal 5/8 to 125 Hz, split into five or fifteen nonoverlapping
250-sample blocks, standardize each channel within each block, and average
block descriptors. No samples are dropped. This differs from the earlier
three-second RQA analysis. Embedding m=3, tau=3 samples, RR=0.05, Theiler=2,
and minimum diagonal/vertical line length=2 are retained. Threshold ties are
retained, exactly as before; consequently actual recurrence rate may exceed
the target, especially for constant or highly quantized signals.

Connectivity is five wPLI views, five pairwise-orthogonalized AEC views and
signed Spearman correlation. Pair ordering is numpy.triu_indices(30, 1).
The connectivity side vector contains 11*435=4785 values. The recurrence
side vector contains 30*5 RQA values (channel-major) then 435 JRS values.
No fit uses another window, participant, label, or dataset-level statistics.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt, welch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


FS_INPUT = 250
FS_SIGNAL = 200
FS_RQA = 125
N_CHANNELS = 30
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
BAND_NAMES = list(BANDS)
CHANNELS = ["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
            "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2"]
RQA_NAMES = ["DET", "LAM", "Lmean", "ENTR", "TT"]
NODE_NAMES = [f"logpower_{band}" for band in BAND_NAMES] + [f"rqa_{name}" for name in RQA_NAMES]
VIEW_NAMES = [f"wpli_{band}" for band in BAND_NAMES] + [f"oaec_{band}" for band in BAND_NAMES] + ["rho_signed", "joint_recurrence"]
IU = np.triu_indices(N_CHANNELS, 1)
M, TAU, TARGET_RR, THEILER, MIN_LINE = 3, 3, 0.05, 2, 2
RQA_BLOCK_SAMPLES = 250
POWER_FLOOR = 1e-30
CONFIG = {
    "input_sampling_hz": FS_INPUT, "signal_sampling_hz": FS_SIGNAL,
    "input_durations_seconds": [10, 30], "resample_signal_ratio": [4, 5],
    "common_bandpass_hz": [1, 45], "butterworth_order": 4,
    "filter_application": "sosfiltfilt independently within each input window",
    "welch_nperseg": 400, "welch_noverlap": 200,
    "power": "log10 mean PSD in half-open frequency bands; floor 1e-30 only for numerical zeros",
    "bands": {name: list(bounds) for name, bounds in BANDS.items()},
    "rqa_sampling_hz": FS_RQA, "resample_rqa_ratio": [5, 8],
    "rqa_block_seconds": 2, "rqa_block_samples": RQA_BLOCK_SAMPLES,
    "rqa_blocks_by_duration": {"10": 5, "30": 15},
    "rqa_embedding_dimension": M, "rqa_delay_samples": TAU,
    "rqa_target_rate": TARGET_RR, "rqa_theiler_samples": THEILER,
    "rqa_minimum_line": MIN_LINE,
    "rqa_standardization": "Within each channel and two-second block: subtract mean, divide by population SD plus 1e-12",
    "rqa_threshold_ties": "Retain distances equal to the quantile threshold",
    "empty_recurrence": "DET and LAM are zero when their recurrence denominator is zero",
    "constant_channel_spearman": "Pairs involving a constant channel have correlation zero",
    "jrs": "Mean(Rc AND Rd) over all embedded matrix entries, divided by target RR; averaged over two-second blocks",
    "channels": CHANNELS, "node_names": NODE_NAMES, "view_names": VIEW_NAMES,
}


def _check_array(name, values, shape=None):
    values = np.asarray(values)
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}; expected {shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    return values


@lru_cache(maxsize=None)
def _sos(low, high):
    return butter(4, [low, high], btype="bandpass", fs=FS_SIGNAL, output="sos")


def preprocess_window(raw250):
    """Return the exact common float32 signal; no per-channel amplitude scaling."""
    raw = _check_array("raw250", raw250)
    if raw.ndim != 2 or raw.shape[0] != N_CHANNELS or raw.shape[1] not in (2500, 7500):
        raise ValueError("Expected 30 channels and 2500 (10 s) or 7500 (30 s) samples at 250 Hz")
    resampled = resample_poly(raw.astype(np.float64), 4, 5, axis=1)
    filtered = sosfiltfilt(_sos(1.0, 45.0), resampled, axis=1)
    output = np.ascontiguousarray(filtered, dtype=np.float32)
    _check_array("processed signal", output)
    return output


def _row_correlation(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return np.sum(a * b, axis=1) / np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1) + 1e-30)


def synchronization_edges(analytic, pair_batch=48):
    """Amplitude-weighted wPLI and symmetrized pairwise oAEC; arrays (435,)."""
    analytic = _check_array("analytic signal", analytic)
    if analytic.ndim != 2 or analytic.shape[0] != N_CHANNELS or pair_batch < 1:
        raise ValueError("Expected a channel-by-time analytic signal and positive pair batch")
    wpli = np.empty(len(IU[0]), dtype=np.float64)
    oaec = np.empty_like(wpli)
    for start in range(0, len(wpli), pair_batch):
        stop = min(start + pair_batch, len(wpli))
        first, second = analytic[IU[0][start:stop]], analytic[IU[1][start:stop]]
        ac, ad = np.abs(first), np.abs(second)
        imaginary = (first * np.conj(second)).imag
        absolute_imaginary = np.abs(imaginary)
        denominator = absolute_imaginary.mean(axis=1)
        wpli[start:stop] = np.divide(np.abs(imaginary.mean(axis=1)), denominator,
                                     out=np.zeros_like(denominator), where=denominator > 0)
        # Magnitudes of the orthogonal components simplify to |Im(zc*zd*)|/|zc|
        # and /|zd|; these equal the complex formulas in the existing extractor.
        second_orthogonal = absolute_imaginary / np.maximum(ac, 1e-20)
        first_orthogonal = absolute_imaginary / np.maximum(ad, 1e-20)
        oaec[start:stop] = 0.5 * (_row_correlation(ac, second_orthogonal) +
                                _row_correlation(ad, first_orthogonal))
    return wpli, oaec


def signed_spearman_edges(signal):
    """Signed rank correlation; zero convention for pairs with a constant channel."""
    ranks = rankdata(signal, axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    sumsquares = np.sum(ranks * ranks, axis=1)
    denominator = np.sqrt(np.outer(sumsquares, sumsquares))
    correlation = np.divide(ranks @ ranks.T, denominator,
                            out=np.zeros_like(denominator), where=denominator > 0)
    return np.clip(correlation[IU], -1.0, 1.0)


@lru_cache(maxsize=None)
def _recurrence_geometry(n):
    i, j = np.indices((n, n))
    eligible = np.abs(i - j) > THEILER
    upper_i, upper_j = np.where(j - i > THEILER)
    diagonal_rows = upper_j - upper_i - THEILER - 1
    return eligible, upper_i, upper_j, diagonal_rows


def recurrence_matrix(values):
    """Embedded Euclidean recurrence with the old quantile/tie/Theiler convention."""
    values = _check_array("recurrence signal", values)
    if values.ndim != 1 or len(values) <= (M - 1) * TAU + THEILER + 1:
        raise ValueError("Recurrence input must be a sufficiently long vector")
    n = len(values) - (M - 1) * TAU
    embedding = np.stack([values[j * TAU:j * TAU + n] for j in range(M)], axis=1)
    distances = squareform(pdist(embedding, metric="euclidean"))
    eligible = _recurrence_geometry(n)[0]
    threshold = np.quantile(distances[eligible], TARGET_RR)
    return (distances <= threshold) & eligible


def _row_run_lengths(matrix):
    """Lengths of every true run along each row, using padded vectorized differences."""
    differences = np.diff(np.pad(matrix.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    start_row, starts = np.where(differences == 1)
    end_row, ends = np.where(differences == -1)
    if not np.array_equal(start_row, end_row):
        raise AssertionError("Unpaired run boundaries")
    return ends - starts


def recurrence_line_lengths(recurrence):
    """Upper diagonal and full vertical runs, numerically matching the earlier RQA."""
    recurrence = np.asarray(recurrence, dtype=bool)
    if recurrence.ndim != 2 or recurrence.shape[0] != recurrence.shape[1]:
        raise ValueError("Recurrence matrix must be square")
    n = len(recurrence)
    if n <= THEILER + 1:
        raise ValueError("Recurrence matrix is too small for the Theiler exclusion")
    _, upper_i, upper_j, rows = _recurrence_geometry(n)
    packed_diagonals = np.zeros((n - THEILER - 1, n), dtype=bool)
    packed_diagonals[rows, upper_i] = recurrence[upper_i, upper_j]
    return _row_run_lengths(packed_diagonals), _row_run_lengths(recurrence.T)


def rqa_metrics(recurrence):
    """DET, LAM, Lmean, ENTR, TT; undefined empty DET/LAM use a documented zero."""
    recurrence = np.asarray(recurrence, dtype=bool)
    diagonal, vertical = recurrence_line_lengths(recurrence)
    upper_i, upper_j = _recurrence_geometry(len(recurrence))[1:3]
    n_upper = recurrence[upper_i, upper_j].sum()
    n_total = recurrence.sum()
    diagonal = diagonal[diagonal >= MIN_LINE]
    vertical = vertical[vertical >= MIN_LINE]
    determinism = diagonal.sum() / n_upper if n_upper else 0.0
    laminarity = vertical.sum() / n_total if n_total else 0.0
    if len(diagonal):
        _, counts = np.unique(diagonal, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        mean_diagonal = diagonal.mean()
    else:
        entropy, mean_diagonal = 0.0, 0.0
    trapping = vertical.mean() if len(vertical) else 0.0
    return np.array([determinism, laminarity, mean_diagonal, entropy, trapping], dtype=np.float64)


def _summarize_rqa125(signal125):
    """Average fixed nonoverlapping two-second blocks, including the entire signal."""
    signal125 = _check_array("125-Hz signal", signal125)
    if signal125.ndim != 2 or signal125.shape[0] != N_CHANNELS or signal125.shape[1] not in (1250, 3750):
        raise ValueError("Expected 30 channels and 1250/3750 samples at 125 Hz")
    n_blocks = signal125.shape[1] // RQA_BLOCK_SAMPLES
    rqa = np.empty((n_blocks, N_CHANNELS, len(RQA_NAMES)), dtype=np.float64)
    jrs = np.empty((n_blocks, len(IU[0])), dtype=np.float64)
    for block in range(n_blocks):
        values = signal125[:, block * RQA_BLOCK_SAMPLES:(block + 1) * RQA_BLOCK_SAMPLES]
        values = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-12)
        matrices = [recurrence_matrix(channel) for channel in values]
        for channel, matrix in enumerate(matrices):
            rqa[block, channel] = rqa_metrics(matrix)
        flattened = np.stack([matrix.ravel() for matrix in matrices]).astype(np.float32)
        # Identical denominator and target-RR scaling to the earlier extractor.
        joint_rate = (flattened @ flattened.T) / flattened.shape[1]
        jrs[block] = (joint_rate / TARGET_RR)[IU]
    return rqa.mean(axis=0), jrs.mean(axis=0)


def recurrence_features(signal200):
    """Return (RQA[30,5], JRS[435]) from the common processed 200-Hz signal."""
    signal200 = _check_array("200-Hz signal", signal200)
    if signal200.shape not in ((30, 2000), (30, 6000)):
        raise ValueError("Expected the common 10- or 30-second signal at 200 Hz")
    signal125 = resample_poly(signal200.astype(np.float64), 5, 8, axis=1)
    return _summarize_rqa125(signal125)


def extract_from_signal(signal200):
    """Extract descriptors from a processed signal; useful for auditing input matching."""
    signal = _check_array("processed signal", signal200)
    if signal.shape not in ((30, 2000), (30, 6000)):
        raise ValueError("Expected 30-channel processed 200-Hz signal of 10 or 30 seconds")
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    values = signal.astype(np.float64)
    frequencies, psd = welch(values, fs=FS_SIGNAL, nperseg=400, noverlap=200, axis=1)
    power = np.stack([np.log10(np.maximum(psd[:, (frequencies >= lo) & (frequencies < hi)].mean(axis=1), POWER_FLOOR))
                      for lo, hi in BANDS.values()], axis=1)
    wpli, oaec = [], []
    for lo, hi in BANDS.values():
        filtered = sosfiltfilt(_sos(lo, hi), values, axis=1)
        analytic = hilbert(filtered, axis=1)
        w, o = synchronization_edges(analytic)
        wpli.append(w)
        oaec.append(o)
    rqa, jrs = recurrence_features(values)
    X = np.concatenate([power, rqa], axis=1).astype(np.float32)
    E = np.stack(wpli + oaec + [signed_spearman_edges(values), jrs]).astype(np.float32)
    result = {
        "signal": signal, "X": X, "E": E,
        "connectivity": E[:11].reshape(-1).copy(),
        "recurrence": np.concatenate([X[:, 5:].reshape(-1), E[11]]),
        "logpower": X[:, :5].reshape(-1).copy(),
        "full_features": np.concatenate([X.reshape(-1), E.reshape(-1)]),
        "metadata": {"duration_seconds": signal.shape[1] // FS_SIGNAL,
                     "signal_sampling_hz": FS_SIGNAL, "rqa_block_seconds": 2,
                     "rqa_n_subwindows": signal.shape[1] // (2 * FS_SIGNAL),
                     "n_channels": N_CHANNELS, "n_unique_features": 5520,
                     "zero_variance_channels": int(np.sum(values.std(axis=1) == 0))},
    }
    for name, value in result.items():
        if name != "metadata":
            _check_array(name, value)
    return result


def process_window(raw250):
    """Process one independent input window and return matched raw/feature branches."""
    return extract_from_signal(preprocess_window(raw250))
