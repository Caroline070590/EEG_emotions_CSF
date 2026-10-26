"""Raw SEED-VII inputs and strictly training-fitted participant-holdout utilities.

``load_dataset(cache)`` returns raw node features X and upper-triangle edges E.
``make_participant_folds(data)`` yields (train, test) row-index arrays.
``fit_fold_scaler(X, E, train).transform(X, E)`` never estimates test statistics.
``model_data(data, scaler)`` reconstructs the symmetric adjacency tensors used by
the existing GNN. Ranking perturbations must change the actual X/A tensors so
that both message-passing and direct-summary branches receive the perturbation.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
EMOTIONS = ["anger", "disgust", "fear", "sadness", "neutral", "happiness", "surprise"]
CHANNELS = ["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
            "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2"]
RQA_NAMES = ["DET", "LAM", "Lmean", "ENTR", "TT"]
NODE_NAMES = [f"logpower_{band}" for band in BANDS] + [f"rqa_{name}" for name in RQA_NAMES]
VIEW_NAMES = [f"wpli_{band}" for band in BANDS] + [f"oaec_{band}" for band in BANDS] + ["rho_signed", "joint_recurrence"]
IU = np.triu_indices(30, 1)


def original_video_to_class(data):
    """Return the verified SEED-VII clip-category map (video IDs 0..79)."""
    mapping = np.asarray(data["video_to_class"], dtype=np.int64)
    if mapping.shape != (80,) or not np.array_equal(np.unique(mapping), np.arange(7)):
        raise ValueError("Expected the SEED-VII 80-clip, seven-condition map")
    return mapping.copy()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(name, values):
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")


def load_dataset(cache):
    """Read 20 x 80 raw trials and verify features, trial metadata and clip metadata."""
    cache = Path(cache)
    filenames = ("features_scalar.csv", "features_arrays.npz", "recurrence_features.npz", "meta.csv", "clips.csv")
    files = {name: cache / name for name in filenames}
    metadata_columns = ["idx", "subject", "video", "clip", "session", "emotion", "title", "source"]
    meta = pd.read_csv(files["meta.csv"])
    feature_meta = pd.read_csv(files["features_scalar.csv"], usecols=metadata_columns)
    clips = pd.read_csv(files["clips.csv"]).sort_values("clip").reset_index(drop=True)
    if not feature_meta[metadata_columns].equals(meta[metadata_columns]):
        raise ValueError("Feature rows disagree with meta.csv on identity, condition, session or source")
    idx = meta["idx"].to_numpy(dtype=np.int64)
    subject = meta["subject"].to_numpy(dtype=np.int64)
    video = meta["video"].to_numpy(dtype=np.int64)
    clip = meta["clip"].to_numpy(dtype=np.int64)
    session = meta["session"].to_numpy(dtype=np.int64)
    if len(meta) != 1600 or not np.array_equal(idx, np.arange(1600)):
        raise ValueError("Cache rows must be indexed 0..1599 in stored array order")
    if not np.array_equal(np.unique(subject), np.arange(1, 21)):
        raise ValueError("Expected original SEED-VII participant IDs 1..20")
    if not np.array_equal(np.unique(video), np.arange(80)) or not np.array_equal(clip, video + 1):
        raise ValueError("Expected videos 0..79 and corresponding clips 1..80")
    if len(set(zip(subject.tolist(), video.tolist()))) != 1600:
        raise ValueError("Participant-clip keys are duplicated or incomplete")
    if len(clips) != 80 or not np.array_equal(clips["clip"].to_numpy(), np.arange(1, 81)):
        raise ValueError("clips.csv must contain clips 1..80 exactly once")
    if not np.array_equal(np.unique(session), np.arange(1, 5)):
        raise ValueError("Expected four SEED-VII sessions")
    # Repeat the clip metadata by the verified zero-based video ID; this catches
    # both incorrect joins and inconsistent labels between participants.
    for name in ["clip", "session", "emotion", "title", "source"]:
        if not np.array_equal(meta[name].to_numpy(), clips[name].to_numpy()[video]):
            raise ValueError(f"Trial metadata disagree with clips.csv for {name}")
    if meta[metadata_columns].isna().any().any():
        raise ValueError("Missing trial metadata")
    label_to_class = {name: i for i, name in enumerate(EMOTIONS)}
    encoded = meta["emotion"].map(label_to_class)
    if encoded.isna().any():
        raise ValueError("Unexpected SEED-VII condition")
    y = encoded.to_numpy(dtype=np.int64)
    video_to_class = clips["emotion"].map(label_to_class).to_numpy(dtype=np.int64)
    if not np.array_equal(np.bincount(video_to_class, minlength=7), [12, 12, 12, 12, 8, 12, 12]):
        raise ValueError("Unexpected clip counts for the seven conditions")
    block_string = (clips["source"] + "|" + clips["emotion"]).to_numpy(dtype=str)
    block_names, video_to_block = np.unique(block_string, return_inverse=True)
    block_labels = np.array([video_to_class[np.flatnonzero(video_to_block == b)[0]]
                             for b in range(len(block_names))], dtype=np.int64)
    if clips["source"].nunique() != 54 or len(block_names) != 56:
        raise ValueError("Expected 54 source programmes and 56 source-by-condition blocks")
    if not np.array_equal(block_labels[video_to_block], video_to_class):
        raise ValueError("A source-by-condition block has inconsistent labels")

    with np.load(files["features_arrays.npz"], allow_pickle=False) as arrays, \
            np.load(files["recurrence_features.npz"], allow_pickle=False) as recurrence:
        if recurrence["rqa_names"].tolist() != RQA_NAMES:
            raise ValueError("Unexpected recurrence feature order")
        power = np.stack([arrays[f"logpow_{band}"] for band in BANDS], axis=-1)
        X = np.concatenate([power, recurrence["rqa"]], axis=-1).astype(np.float32)
        edge_keys = [f"pair_{name}" for name in VIEW_NAMES[:10]] + ["rho_signed"]
        E = np.stack([arrays[key] for key in edge_keys] + [recurrence["jrs"]], axis=1).astype(np.float32)
    if X.shape != (1600, 30, 10) or E.shape != (1600, 12, 435):
        raise ValueError(f"Unexpected input dimensions: X={X.shape}, E={E.shape}")
    _finite("Node features", X)
    _finite("Edge features", E)
    feature_names = [f"{feature}:{channel}" for channel in CHANNELS for feature in NODE_NAMES]
    feature_names += [f"{view}:{CHANNELS[a]}--{CHANNELS[b]}"
                      for view in VIEW_NAMES for a, b in zip(*IU)]
    if len(feature_names) != 5520 or len(set(feature_names)) != 5520:
        raise AssertionError("The unique scalar feature manifest is inconsistent")
    mixed = clips.groupby("source")["emotion"].nunique()
    mixed_names = mixed[mixed > 1].index.tolist()
    session_counts = pd.crosstab(clips["session"], clips["emotion"]).reindex(
        index=range(1, 5), columns=EMOTIONS, fill_value=0)
    manifest = {
        "dataset": "SEED-VII", "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path),
                                                   "bytes": path.stat().st_size} for name, path in files.items()},
        "n_trials": 1600, "n_participants": 20, "n_videos": 80, "n_classes": 7,
        "n_sessions": 4, "n_source_programmes": 54, "n_source_condition_blocks": 56,
        "node_shape": list(X.shape), "edge_shape": list(E.shape), "n_unique_features": len(feature_names),
        "participant_internal_ids": [1, 20], "participant_display_ids": [1, 20],
        "emotions": EMOTIONS, "class_trial_counts": np.bincount(y, minlength=7).tolist(),
        "video_to_class": video_to_class.tolist(), "video_to_block": video_to_block.tolist(),
        "video_to_session": clips["session"].to_numpy(dtype=int).tolist(),
        "block_names": block_names.tolist(), "block_labels": block_labels.tolist(),
        "block_class_counts": np.bincount(block_labels, minlength=7).tolist(),
        "mixed_source_programmes": mixed_names,
        "session_by_class_clip_counts": session_counts.to_numpy(dtype=int).tolist(),
        "channels": CHANNELS, "node_names": NODE_NAMES, "view_names": VIEW_NAMES,
        "source_epoch": "Official preprocessed 62-channel EEG at 200 Hz; same 30 FACED sensors; last 30 seconds resampled by 5/4 to 250 Hz (7500 samples)",
        "preprocessing": "Raw cached features; scaling is fitted separately on training participants only",
        "null": "Labels shuffled across 56 source-by-original-condition blocks, shared by every participant; block-class counts preserved, clip-class counts may change",
        "scope": "Unseen participants viewing the familiar clips and sessions; sessions differ in condition composition",
    }
    return dict(X=X, E=E, y=y, subject=subject, video=video, idx=idx, clip=clip,
                session=session, source=meta["source"].to_numpy(dtype=str),
                source_condition_block=video_to_block[video], video_to_class=video_to_class,
                video_to_block=video_to_block, video_to_session=clips["session"].to_numpy(dtype=np.int64),
                block_names=block_names, block_labels=block_labels,
                node_names=list(NODE_NAMES), view_names=list(VIEW_NAMES), channels=list(CHANNELS),
                feature_names=feature_names, manifest=manifest)


def make_participant_folds(data, n_splits=5, seed=20261024):
    """Fixed shuffled participant folds, each trial held out exactly once."""
    subject = np.asarray(data["subject"])
    participants = np.unique(subject)
    if n_splits < 2 or n_splits > len(participants):
        raise ValueError("n_splits must be between 2 and the number of participants")
    shuffled = np.random.default_rng(seed).permutation(participants)
    folds, coverage = [], np.zeros(len(subject), dtype=np.int64)
    for held in np.array_split(shuffled, n_splits):
        is_test = np.isin(subject, held)
        train, test = np.flatnonzero(~is_test), np.flatnonzero(is_test)
        if np.intersect1d(subject[train], subject[test]).size:
            raise AssertionError("Participants overlap training and test sets")
        coverage[test] += 1
        folds.append((train, test))
    if not np.all(coverage == 1):
        raise AssertionError("Not every trial is held out exactly once")
    return folds


@dataclass
class FoldScaler:
    """Per-scalar training means/SDs; nearly constant training columns become zero."""
    x_mean: np.ndarray
    x_std: np.ndarray
    x_constant: np.ndarray
    e_mean: np.ndarray
    e_std: np.ndarray
    e_constant: np.ndarray
    n_training_rows: int
    threshold: float = 1e-6

    def transform(self, X, E):
        X, E = np.asarray(X), np.asarray(E)
        if X.shape[1:] != self.x_mean.shape or E.shape[1:] != self.e_mean.shape:
            raise ValueError("Input feature dimensions differ from the fitted scaler")
        _finite("Node features", X)
        _finite("Edge features", E)
        scaled_x = ((X.astype(np.float64) - self.x_mean) / self.x_std).astype(np.float32)
        scaled_e = ((E.astype(np.float64) - self.e_mean) / self.e_std).astype(np.float32)
        scaled_x[:, self.x_constant] = 0.0
        scaled_e[:, self.e_constant] = 0.0
        _finite("Scaled nodes", scaled_x)
        _finite("Scaled edges", scaled_e)
        return np.ascontiguousarray(scaled_x), np.ascontiguousarray(scaled_e)

    def to_dict(self):
        return {"n_training_rows": self.n_training_rows, "constant_threshold": self.threshold,
                "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "x_constant": self.x_constant.tolist(), "e_mean": self.e_mean.tolist(),
                "e_std": self.e_std.tolist(), "e_constant": self.e_constant.tolist()}


def fit_fold_scaler(X, E, train_idx, threshold=1e-6):
    """Fit only train_idx; held-out values cannot affect any fitted statistic."""
    X, E, train = np.asarray(X), np.asarray(E), np.asarray(train_idx, dtype=np.int64)
    if len(X) != len(E) or train.ndim != 1 or not len(train):
        raise ValueError("Training rows must be a nonempty one-dimensional index vector")
    if len(np.unique(train)) != len(train) or train.min() < 0 or train.max() >= len(X):
        raise ValueError("Training rows contain duplicates or invalid indices")
    if threshold <= 0:
        raise ValueError("Constant threshold must be positive")
    moments = []
    for name, values in (("Training nodes", X[train]), ("Training edges", E[train])):
        _finite(name, values)
        values = values.astype(np.float64)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
        constant = std < threshold
        std[constant] = 1.0
        moments.extend([mean, std, constant])
    return FoldScaler(*moments, n_training_rows=len(train), threshold=float(threshold))


def edges_to_adjacency(E):
    """Expand (...,435) upper triangles to symmetric (...,30,30), zero diagonal."""
    E = np.asarray(E)
    if E.shape[-1] != 435:
        raise ValueError("Expected 435 upper-triangle edges for 30 sensors")
    _finite("Edge features", E)
    A = np.zeros(E.shape[:-1] + (30, 30), dtype=np.float32)
    A[..., IU[0], IU[1]] = E
    A[..., IU[1], IU[0]] = E
    return A


def model_data(data, scaler):
    """Adapt raw data to the existing GNN's data dictionary without fitting anything."""
    X, E = scaler.transform(data["X"], data["E"])
    return dict(X=X, A=edges_to_adjacency(E), y=data["y"], subject=data["subject"],
                video=data["video"], views=list(data["view_names"]))


def ranking_groups(data):
    """22 atomic feature types and four nonduplicate family groups (26 total)."""
    groups = [dict(name=name, kind="atomic", node_indices=[i], edge_indices=[])
              for i, name in enumerate(data["node_names"])]
    groups += [dict(name=name, kind="atomic", node_indices=[], edge_indices=[i])
               for i, name in enumerate(data["view_names"])]
    groups += [dict(name="power_all", kind="family", node_indices=list(range(5)), edge_indices=[]),
               dict(name="rqa_all", kind="family", node_indices=list(range(5, 10)), edge_indices=[]),
               dict(name="wpli_all", kind="family", node_indices=[], edge_indices=list(range(5))),
               dict(name="oaec_all", kind="family", node_indices=[], edge_indices=list(range(5, 10)))]
    if len(groups) != 26 or len({group["name"] for group in groups}) != 26:
        raise ValueError("Ranking groups require the fixed 10-node-feature/12-view schema")
    return groups


def video_block_permutation(subject, video, test_idx, seed):
    """Global donor rows aligned with test_idx, retaining each test participant.

    A single randomly permuted video order is shared by every held-out
    participant. Each participant must have one test row for every video.
    Reuse returned donors across feature groups within the same repetition.
    """
    subject, video = np.asarray(subject), np.asarray(video)
    test = np.asarray(test_idx, dtype=np.int64)
    if len(subject) != len(video) or test.ndim != 1 or not len(test):
        raise ValueError("Invalid subject/video arrays or empty test indices")
    if len(np.unique(test)) != len(test) or test.min() < 0 or test.max() >= len(subject):
        raise ValueError("Duplicate or invalid test indices")
    videos = np.unique(video)
    shuffled = np.random.default_rng(seed).permutation(videos)
    video_map = dict(zip(videos.tolist(), shuffled.tolist()))
    lookup = {(int(subject[row]), int(video[row])): int(row) for row in test}
    expected = len(np.unique(subject[test])) * len(videos)
    if len(lookup) != len(test) or len(test) != expected:
        raise ValueError("Every held-out participant must have every video exactly once")
    donors = np.array([lookup[(int(subject[row]), video_map[int(video[row])])] for row in test], dtype=np.int64)
    if not np.array_equal(subject[donors], subject[test]):
        raise AssertionError("Block permutation changed participant identity")
    return donors


def permuted_video_labels(data, seed):
    """Shuffle 56 source-by-condition block labels; return (trial labels, clip map).

    One label is assigned to every excerpt in a block and shared by all
    participants. Two programmes spanning two original conditions contribute
    two blocks each. Block label counts are preserved; clip counts need not be.
    """
    block_labels = np.asarray(data["block_labels"], dtype=np.int64)
    video_to_block = np.asarray(data["video_to_block"], dtype=np.int64)
    video = np.asarray(data["video"], dtype=np.int64)
    if block_labels.shape != (56,) or video_to_block.shape != (80,):
        raise ValueError("Expected 56 source-condition blocks mapping 80 clips")
    if video_to_block.min() < 0 or video_to_block.max() >= len(block_labels):
        raise ValueError("Invalid clip-to-block map")
    if video.ndim != 1 or not len(video) or video.min() < 0 or video.max() >= 80:
        raise ValueError("Video IDs must lie in 0..79")
    shuffled_labels = np.random.default_rng(seed).permutation(block_labels)
    mapping = shuffled_labels[video_to_block]
    return mapping[video], mapping
