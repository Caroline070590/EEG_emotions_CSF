"""SEED-VII session-only diagnostic fitted on outer-training participants.

No EEG enters this classifier. The decision rule maximizes estimated
P(session | class), the class-balanced objective, with fixed class-order ties.
It diagnoses familiar recording-session information; it is not an emotion model.
"""
import argparse
import json
import os
from pathlib import Path
import uuid

import numpy as np

from common import PRIMARY_SEED, atomic_json, digest, fingerprint, participant_split


def fit_predict_session(session, y, train, test, n_classes):
    session, y = np.asarray(session), np.asarray(y)
    train, test = np.asarray(train), np.asarray(test)
    if np.intersect1d(train, test).size:
        raise ValueError("Session control training and test rows overlap")
    counts = np.bincount(y[train], minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        raise ValueError("Every declared class must occur in training")
    pred = np.empty(len(test), dtype=np.int64)
    for value in np.unique(session[test]):
        same = train[session[train] == value]
        if not len(same):
            raise ValueError("A test session is absent from training")
        frequencies = np.bincount(y[same], minlength=n_classes) / counts
        pred[session[test] == value] = int(np.argmax(frequencies))
    return pred


def session_predictions(meta, n_classes, fold_seed=PRIMARY_SEED, n_folds=5):
    if n_folds != 5:
        raise ValueError("The matched benchmark uses five participant folds")
    n = len(meta["y"])
    pred, fold, coverage = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    roles = []
    for index in range(n_folds):
        split = participant_split(meta["subject"], index, fold_seed)
        train, test = split["outer_train"], split["test"]
        pred[test] = fit_predict_session(meta["session"], meta["y"], train, test, n_classes)
        fold[test] = index; coverage[test] += 1
        roles.append({"fold": index, **{name + "_subjects": np.unique(meta["subject"][rows]).tolist() for name, rows in split.items()}})
    if np.any(coverage != 1) or np.any(pred < 0):
        raise ValueError("Incomplete session-control OOF coverage")
    return pred, fold, roles


def build_control(prepared, out):
    prepared, out = Path(prepared), Path(out)
    config = json.loads((out / "config.json").read_text())
    config_hash = fingerprint(config)
    source_hash = digest(Path(__file__))
    if config["source_sha256"]["session_control.py"] != source_hash:
        raise ValueError("Session-control implementation differs from the frozen configuration")
    metadata_path = prepared / "seed7" / "30s" / "meta.npz"
    metadata_hash = digest(metadata_path)
    expected = config["prepared_manifest"]["seed7"]["30"]["output_sha256"]["meta.npz"]
    if metadata_hash != expected:
        raise ValueError("Session-control metadata differ from the frozen prepared input")
    with np.load(metadata_path, allow_pickle=False) as saved:
        keys = ("y", "trial_idx", "subject", "video", "window_id", "window_start", "window_idx", "session")
        meta = {key: np.asarray(saved[key]) for key in keys}
    spec = config["datasets"]["seed7"]
    n_classes = len(spec["class_names"])
    n_trials = int(spec["n_subjects"]) * int(spec["n_videos"])
    if any(value.ndim != 1 or len(value) != n_trials or not np.issubdtype(value.dtype, np.integer) for value in meta.values()):
        raise ValueError("Session-control metadata must contain complete integer trial vectors")
    if len(np.unique(meta["trial_idx"])) != n_trials or len(set(zip(meta["subject"], meta["video"]))) != n_trials:
        raise ValueError("Duplicate session-control trials")
    if np.any(meta["window_id"] != 0) or np.any(meta["window_start"] != 0):
        raise ValueError("Session control must use the one-window-per-trial 30-s metadata")
    if len(np.unique(meta["subject"])) != spec["n_subjects"] or len(np.unique(meta["video"])) != spec["n_videos"]:
        raise ValueError("Session-control participant/video population differs from configuration")
    if meta["y"].min() < 0 or meta["y"].max() >= n_classes:
        raise ValueError("Invalid session-control class labels")
    for video in np.unique(meta["video"]):
        take = meta["video"] == video
        if len(np.unique(meta["y"][take])) != 1 or len(np.unique(meta["session"][take])) != 1:
            raise ValueError("A familiar video's condition/session differs across participants")
    pred, fold, roles = session_predictions(meta, n_classes, int(config.get("primary_seed", PRIMARY_SEED)), int(config["n_folds"]))
    json_path = out / "session_control.json"
    if json_path.exists() and json.loads(json_path.read_text())["config_sha256"] != config_hash:
        raise ValueError("Refusing to overwrite session control from another configuration")
    prediction_path = out / "session_control.npz"
    temporary = prediction_path.with_name("session_control." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(temporary, **meta, pred=pred, fold=fold, probabilities=np.eye(n_classes, dtype=np.float32)[pred])
    os.replace(temporary, prediction_path)
    accuracy = float(np.mean([np.mean(pred[meta["y"] == cls] == cls) for cls in range(n_classes)]))
    result = {"complete": True, "dataset": "seed7", "n_classes": n_classes,
              "definition": "Outer-training class-balanced session classifier; fixed class-order ties; no EEG input",
              "scope": "Unseen participants viewing familiar clips in familiar recording sessions. This is a session-design diagnostic, not a label-permutation test or evidence of emotion-specific information.",
              "duration_application": "One session prediction per trial applies identically to the three-window average and the single 30-s window",
              "balanced_accuracy": accuracy, "fold_roles": roles, "config_sha256": config_hash,
              "metadata_sha256": metadata_hash, "source_sha256": source_hash,
              "predictions_path": prediction_path.name, "predictions_sha256": digest(prediction_path)}
    atomic_json(json_path, result)
    print(json.dumps({"session_only_balanced_accuracy_percent": 100 * accuracy, "n_trials": n_trials}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_control(args.prepared, args.out)
