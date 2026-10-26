"""One participant-held-out EEGNet fold, with inner epoch selection and refit.

Production requires a GPU and the frozen 300-task configuration. The outer and
inner participant splits always use PRIMARY_SEED, independently of training
seed. Normalization and inverse-frequency class weights are fitted first on
inner training only, then freshly on all outer training. Validation uses trial
balanced accuracy and unweighted trial cross-entropy to break ties; the 10-s
windows are probability-averaged. At least 20 selection epochs are run, but an
earlier best checkpoint can determine the fresh outer refit's epoch count.

The shuffled control permutes complete trials within each participant, preserving
window position. Training uses a new permutation each epoch; validation and
test use different fixed permutations. All donor arrays and their seeds are
saved. ``--smoke --device cpu`` is a software exercise: it selects two people
per role and one whole trial per class/person, runs at most two epochs, writes
only smoke-prefixed paths, and never declares a completed production result.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from common import (DATASETS, HERE, MODELS, PRIMARY_SEED, SEEDS, atomic_json,
                    balanced_accuracy, digest, fingerprint, participant_split,
                    side_donors, trial_probabilities)
from models import build_model


VALIDATION_CRITERION = "trial balanced accuracy; lower validation loss breaks ties"
TRAINING_DEFAULTS = {"max_epochs": 100, "min_epochs": 20, "patience": 20,
                     "learning_rate": 0.001, "batch_size": 64}
REQUIRED_SOURCES = {"common.py", "features.py", "prepare.py", "models.py", "train.py"}
SIDE_SLICES = {"eegnet": None, "eegnet_connectivity": slice(0, 4785),
               "eegnet_recurrence": slice(4785, 5370),
               "eegnet_combined": slice(0, 5370), "eegnet_shuffled": slice(0, 5370)}


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def validate_config(config, task_index, smoke=False):
    sources = config.get("source_sha256", {})
    if not REQUIRED_SOURCES.issubset(sources):
        raise ValueError("Configuration does not pin all required source hashes")
    for relative, expected in sources.items():
        path = (HERE / relative).resolve()
        if HERE not in path.parents or not path.is_file() or digest(path) != expected:
            raise ValueError("Frozen source hash mismatch: " + relative)
    if config.get("primary_seed") != PRIMARY_SEED or config.get("n_folds") != 5:
        raise ValueError("The primary split seed and five folds are fixed")
    tasks = config.get("tasks", [])
    if task_index not in range(len(tasks)):
        raise ValueError("Task index is outside config['tasks']")
    identities = [tuple(t[k] for k in ("dataset", "seconds", "model", "seed", "fold")) for t in tasks]
    if len(set(identities)) != len(identities):
        raise ValueError("Configuration has duplicate task identities")
    if not smoke:
        expected = set(itertools.product(DATASETS, (10, 30), MODELS, SEEDS, range(5)))
        if len(tasks) != 300 or set(identities) != expected:
            raise ValueError("Production requires the fixed 300 matched tasks")
        if (config.get("datasets") != DATASETS or set(config.get("models", [])) != set(MODELS)
                or set(config.get("seeds", [])) != set(SEEDS)
                or set(config.get("seconds", [])) != {10, 30}):
            raise ValueError("Production dataset/model/seed/window specification changed")
    training = dict(TRAINING_DEFAULTS, **config.get("training", {}))
    if not smoke and any(training[k] != v for k, v in TRAINING_DEFAULTS.items()):
        raise ValueError("Production training settings differ from the frozen protocol")
    task = tasks[task_index]
    if (task["dataset"] not in config["datasets"] or task["seconds"] not in (10, 30)
            or task["model"] not in MODELS or task["fold"] not in range(5)):
        raise ValueError("Invalid task specification")
    return task, training


def load_prepared(prepared, task, config, smoke=False):
    name, seconds = task["dataset"], task["seconds"]
    directory = Path(prepared) / name / f"{seconds}s"
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    pinned = config["prepared_manifest"][name][str(seconds)]
    if manifest["output_sha256"] != pinned["output_sha256"]:
        raise ValueError("Prepared output hashes differ from the frozen manifest")
    manifest_hash = digest(manifest_path)
    expected_manifest = pinned.get("manifest_sha256", pinned.get("sha256"))
    if expected_manifest is not None and manifest_hash != expected_manifest:
        raise ValueError("Prepared manifest hash changed")
    if digest(directory / "meta.npz") != pinned["output_sha256"]["meta.npz"]:
        raise ValueError("Prepared metadata hash changed")
    if manifest["dataset"] != name or manifest["seconds"] != seconds:
        raise ValueError("Prepared manifest dataset/duration mismatch")
    with np.load(directory / "meta.npz", allow_pickle=False) as saved:
        meta = {k: saved[k] for k in saved.files}
    required = ("y", "subject", "video", "trial_idx", "window_idx", "window_id", "window_start")
    if not set(required).issubset(meta):
        raise ValueError("Prepared metadata is missing required window identifiers")
    n = len(meta["subject"])
    for key in required:
        value = meta[key]
        if value.shape != (n,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Prepared metadata must contain integer vectors: " + key)
    if not np.array_equal(meta["window_idx"], np.arange(n)):
        raise ValueError("Prepared window_idx must equal memmap row indices")
    signal = np.load(directory / "signal.npy", mmap_mode="r", allow_pickle=False)
    side = np.load(directory / "side.npy", mmap_mode="r", allow_pickle=False)
    if signal.shape != (n, 30, seconds * 200) or side.shape != (n, 5370):
        raise ValueError("Prepared signal or side-array shape mismatch")
    if signal.dtype != np.float32 or side.dtype != np.float32:
        raise ValueError("Prepared arrays must be float32")
    dataset = config["datasets"][name]
    n_classes = len(dataset["class_names"])
    if not np.array_equal(np.unique(meta["y"]), np.arange(n_classes)):
        raise ValueError("Prepared labels do not cover the declared contiguous classes")
    people, videos = np.unique(meta["subject"]), np.unique(meta["video"])
    if len(people) != dataset["n_subjects"] or len(videos) != dataset["n_videos"]:
        raise ValueError("Prepared participant/video counts differ from configuration")
    n_windows = 30 // seconds
    if n != len(people) * len(videos) * n_windows:
        raise ValueError("Prepared design is not complete participants by videos by windows")
    pairs = set()
    for trial in np.unique(meta["trial_idx"]):
        rows = np.flatnonzero(meta["trial_idx"] == trial)
        for key in ("subject", "video", "y"):
            if len(np.unique(meta[key][rows])) != 1:
                raise ValueError("A trial has inconsistent participant, video, or class metadata")
        order = rows[np.argsort(meta["window_id"][rows])]
        if (not np.array_equal(meta["window_id"][order], np.arange(n_windows))
                or not np.array_equal(meta["window_start"][order], seconds * np.arange(n_windows))):
            raise ValueError("A trial does not contain its complete nonoverlapping windows")
        pair = (int(meta["subject"][rows[0]]), int(meta["video"][rows[0]]))
        if pair in pairs:
            raise ValueError("Repeated participant-video trial")
        pairs.add(pair)
    for video in videos:
        if len(np.unique(meta["y"][meta["video"] == video])) != 1:
            raise ValueError("A video's labels differ across participants")
    return signal, side, meta, manifest_hash


def smoke_splits(meta, splits):
    """Two people per role, one complete trial per class/person; no cut windows."""
    result = {}
    for role in ("inner_train", "validation", "test"):
        rows = splits[role]
        selected = []
        for person in np.unique(meta["subject"][rows])[:2]:
            person_rows = rows[meta["subject"][rows] == person]
            for label in np.unique(meta["y"]):
                candidates = person_rows[meta["y"][person_rows] == label]
                if not len(candidates):
                    raise ValueError("Smoke participant is missing a class")
                trial = np.min(meta["trial_idx"][candidates])
                selected.extend(candidates[meta["trial_idx"][candidates] == trial].tolist())
        result[role] = np.array(sorted(selected), dtype=np.int64)
    result["outer_train"] = np.sort(np.r_[result["inner_train"], result["validation"]])
    return result


def fit_scalers(signal, side, rows, chunk_size=16):
    """Stable float64 population moments, fitted exclusively to supplied rows."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        raise ValueError("Cannot fit scalers without training windows")
    statistics = {}
    for name, array, axes, width in (("raw", signal, (0, 2), signal.shape[1]),
                                     ("side", side, (0,), side.shape[1])):
        count = 0; mean = np.zeros(width); m2 = np.zeros(width)
        for start in range(0, len(rows), chunk_size):
            values = np.asarray(array[rows[start:start + chunk_size]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite training input")
            n = values.shape[0] * (values.shape[2] if name == "raw" else 1)
            batch_mean = values.mean(axis=axes)
            batch_m2 = values.var(axis=axes) * n
            delta = batch_mean - mean
            m2 += batch_m2 + delta * delta * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        std = np.sqrt(np.maximum(m2 / count, 0))
        constant = std <= 1e-12
        std[constant] = 1
        statistics[name + "_mean"] = mean.astype(np.float32)
        statistics[name + "_std"] = std.astype(np.float32)
        statistics[name + "_constant"] = constant
        statistics[name + "_count"] = np.array(count, dtype=np.int64)
    return statistics


def class_weights(y, rows, n_classes):
    counts = np.bincount(y[rows], minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        raise ValueError("Every training split must contain all classes")
    return (len(rows) / (n_classes * counts)).astype(np.float32)


def make_batch(signal, side, meta, rows, scalers, variant, device, donors=None):
    raw = np.array(signal[rows], dtype=np.float32, copy=True)
    raw -= scalers["raw_mean"][None, :, None]
    raw /= scalers["raw_std"][None, :, None]
    raw[:, scalers["raw_constant"], :] = 0
    features = None
    selected = SIDE_SLICES[variant]
    if selected is not None:
        source_rows = rows if donors is None else donors[rows]
        if np.any(source_rows < 0):
            raise ValueError("Missing side-feature donors")
        features = np.array(side[source_rows, selected], dtype=np.float32, copy=True)
        features -= scalers["side_mean"][selected]
        features /= scalers["side_std"][selected]
        features[:, scalers["side_constant"][selected]] = 0
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite normalized side features")
        features = torch.from_numpy(features).to(device)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite normalized raw signal")
    return (torch.from_numpy(raw).to(device), features,
            torch.as_tensor(meta["y"][rows], dtype=torch.long, device=device))


def train_epoch(model, optimizer, signal, side, meta, rows, scalers,
                variant, device, weights, batch_size, order_seed, donors=None):
    model.train()
    order = np.random.default_rng(order_seed).permutation(rows)
    total, denominator = 0.0, 0.0
    for start in range(0, len(order), batch_size):
        take = order[start:start + batch_size]
        raw, features, y = make_batch(signal, side, meta, take, scalers, variant, device, donors)
        optimizer.zero_grad(set_to_none=True)
        losses = F.cross_entropy(model(raw, features), y, weight=weights, reduction="none")
        normalizer = weights[y].sum()
        loss = losses.sum() / normalizer
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward(); optimizer.step(); model.apply_constraints()
        total += float(losses.detach().sum().cpu())
        denominator += float(normalizer.detach().cpu())
    return total / denominator


@torch.no_grad()
def predict(model, signal, side, meta, rows, scalers, variant, device,
            batch_size, donors=None):
    model.eval(); output = []
    for start in range(0, len(rows), batch_size):
        raw, features, _ = make_batch(signal, side, meta, rows[start:start + batch_size],
                                      scalers, variant, device, donors)
        output.append(model(raw, features).softmax(dim=1).cpu().numpy())
    probabilities = np.concatenate(output)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite predicted probabilities")
    return probabilities


def trial_metrics(probabilities, meta, rows):
    probability, y, _ = trial_probabilities(probabilities, meta["trial_idx"][rows], meta["y"][rows])
    if np.any(np.bincount(y, minlength=probability.shape[1]) == 0):
        raise ValueError("Evaluated trials must contain every class")
    loss = float(-np.log(np.maximum(probability[np.arange(len(y)), y], 1e-30)).mean())
    return balanced_accuracy(y, probability), loss


def improves(accuracy, loss, best_accuracy, best_loss):
    return accuracy > best_accuracy + 1e-12 or (abs(accuracy - best_accuracy) <= 1e-12 and loss < best_loss - 1e-12)


def backbone_fingerprint(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("side_"):
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def _checkpoint(model, optimizer, config_hash, task, epoch, stage, initial_hash):
    return {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "config_sha256": config_hash,
            "task": task, "epoch": epoch, "stage": stage,
            "initial_backbone_sha256": initial_hash}


def run_task(args):
    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    task, training = validate_config(config, args.task, args.smoke)
    device = torch.device(args.device)
    if device.type != "cuda" and not args.smoke:
        raise ValueError("Production training requires CUDA; CPU is available only with --smoke")
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested CPU/CUDA device is unavailable")
    if args.smoke_epochs not in (1, 2):
        raise ValueError("Smoke runs use one or two epochs only")
    torch.set_num_threads(max(1, args.threads))
    config_hash = fingerprint(config)
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    config_copy = root / ("smoke_config.json" if args.smoke else "config.json")
    if config_copy.exists() and fingerprint(json.loads(config_copy.read_text())) != config_hash:
        raise ValueError("Result directory already contains a different configuration")
    atomic_json(config_copy, config)
    prefix = "smoke_" if args.smoke else ""
    stem = f"{task['dataset']}_{task['seconds']}s_{task['model']}_seed{task['seed']}_fold{task['fold']}"
    stem = prefix + stem
    paths = {kind: root / (prefix + folder) / (stem + suffix) for kind, folder, suffix in (
        ("run", "runs", ".json"), ("predictions", "predictions", ".npz"),
        ("scalers", "scalers", ".npz"), ("splits", "splits", ".npz"),
        ("curves", "curves", ".json"), ("donors", "donors", ".npz"),
        ("selection", "checkpoints", "_selection.pt"), ("refit", "checkpoints", "_refit.pt"))}
    if paths["run"].exists():
        previous = json.loads(paths["run"].read_text())
        if previous.get("config_sha256") != config_hash:
            raise ValueError("Existing fold has different configuration")
        if previous.get("complete") is True or previous.get("smoke_complete") is True:
            for name, expected in previous["artifact_sha256"].items():
                if digest(root / name) != expected:
                    raise ValueError("Completed artifact changed: " + name)
            print("Already complete", paths["run"], flush=True)
            return previous
    signal, side, meta, manifest_hash = load_prepared(args.prepared, task, config, args.smoke)
    splits = participant_split(meta["subject"], task["fold"], PRIMARY_SEED)
    if args.smoke:
        splits = smoke_splits(meta, splits)
        training = dict(training, max_epochs=args.smoke_epochs, min_epochs=1, patience=args.smoke_epochs)
    n_classes = len(config["datasets"][task["dataset"]]["class_names"])
    run_seed = int(task["seed"]) + int(task["fold"])
    roles = {key + "_subjects": np.unique(meta["subject"][splits[role]]).tolist()
             for key, role in (("outer_train", "outer_train"), ("inner_train", "inner_train"),
                               ("validation", "validation"), ("test", "test"))}
    result = dict(task, task_index=args.task, config_sha256=config_hash,
                  source_sha256=config["source_sha256"], prepared_manifest_sha256=manifest_hash,
                  complete=False, production_complete=False, smoke=bool(args.smoke), status="running",
                  n_classes=n_classes, validation_criterion=VALIDATION_CRITERION,
                  split_seed=PRIMARY_SEED, initialization_seed=run_seed, **roles)
    atomic_json(paths["run"], result)
    atomic_npz(paths["splits"], **splits)
    curves = {"selection": [], "refit": []}
    donor_records = {}
    variant, batch_size = task["model"], int(training["batch_size"])
    is_shuffled = variant == "eegnet_shuffled"
    # Distinct deterministic namespaces; no class labels enter donor selection.
    donor_seeds = {"validation": run_seed + 100000, "test": run_seed + 200000,
                   "selection_train_base": run_seed + 300000, "refit_train_base": run_seed + 400000}
    validation_donors = side_donors(meta, splits["validation"], donor_seeds["validation"]) if is_shuffled else None
    test_donors = side_donors(meta, splits["test"], donor_seeds["test"]) if is_shuffled else None
    if is_shuffled:
        donor_records.update(validation_rows=splits["validation"], test_rows=splits["test"],
                             validation_donors=validation_donors[splits["validation"]],
                             test_donors=test_donors[splits["test"]])
    scalers_inner = fit_scalers(signal, side, splits["inner_train"])
    scalers_outer = fit_scalers(signal, side, splits["outer_train"])
    inner_weights = class_weights(meta["y"], splits["inner_train"], n_classes)
    outer_weights = class_weights(meta["y"], splits["outer_train"], n_classes)
    atomic_npz(paths["scalers"], **{"inner_" + k: v for k, v in scalers_inner.items()},
               **{"refit_" + k: v for k, v in scalers_outer.items()},
               inner_class_weights=inner_weights, refit_class_weights=outer_weights)
    seed_everything(run_seed)
    model = build_model(variant, n_classes, task["seconds"] * 200).to(device)
    initial_hash = backbone_fingerprint(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    weights = torch.as_tensor(inner_weights, device=device)
    best_accuracy, best_loss, selected_epoch, stale = -1.0, float("inf"), 0, 0
    selected_donors = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epoch_start = time.monotonic()
        donors = side_donors(meta, splits["inner_train"], donor_seeds["selection_train_base"] + epoch) if is_shuffled else None
        if is_shuffled:
            selected_donors.append(donors[splits["inner_train"]])
        train_loss = train_epoch(model, optimizer, signal, side, meta, splits["inner_train"], scalers_inner,
                                 variant, device, weights, batch_size, run_seed + epoch, donors)
        validation = predict(model, signal, side, meta, splits["validation"], scalers_inner,
                             variant, device, batch_size, validation_donors)
        accuracy, loss = trial_metrics(validation, meta, splits["validation"])
        improved = improves(accuracy, loss, best_accuracy, best_loss)
        if improved:
            best_accuracy, best_loss, selected_epoch, stale = accuracy, loss, epoch, 0
            atomic_checkpoint(paths["selection"], _checkpoint(model, optimizer, config_hash, task,
                                                              epoch, "selection", initial_hash))
        else:
            stale += 1
        row = {"epoch": epoch, "training_weighted_cross_entropy": train_loss,
               "validation_trial_balanced_accuracy": accuracy, "validation_trial_cross_entropy": loss,
               "improved": improved, "seconds": time.monotonic() - epoch_start}
        curves["selection"].append(row); atomic_json(paths["curves"], curves)
        print(json.dumps(dict(task, phase="selection", **row)), flush=True)
        if epoch >= int(training["min_epochs"]) and stale >= int(training["patience"]):
            break
    # Start a fresh model/optimizer from the SAME seed. Never warm-start the refit.
    del optimizer, model
    seed_everything(run_seed)
    model = build_model(variant, n_classes, task["seconds"] * 200).to(device)
    refit_initial_hash = backbone_fingerprint(model)
    if refit_initial_hash != initial_hash:
        raise ValueError("Fresh refit initialization differs from selection initialization")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    weights = torch.as_tensor(outer_weights, device=device)
    refit_donors = []
    for epoch in range(1, selected_epoch + 1):
        epoch_start = time.monotonic()
        donors = side_donors(meta, splits["outer_train"], donor_seeds["refit_train_base"] + epoch) if is_shuffled else None
        if is_shuffled:
            refit_donors.append(donors[splits["outer_train"]])
        loss = train_epoch(model, optimizer, signal, side, meta, splits["outer_train"], scalers_outer,
                           variant, device, weights, batch_size, run_seed + epoch, donors)
        row = {"epoch": epoch, "training_weighted_cross_entropy": loss,
               "seconds": time.monotonic() - epoch_start}
        curves["refit"].append(row); atomic_json(paths["curves"], curves)
        print(json.dumps(dict(task, phase="refit", **row)), flush=True)
    atomic_checkpoint(paths["refit"], _checkpoint(model, optimizer, config_hash, task,
                                                  selected_epoch, "refit", initial_hash))
    prediction = predict(model, signal, side, meta, splits["test"], scalers_outer,
                         variant, device, batch_size, test_donors)
    test_accuracy, test_loss = trial_metrics(prediction, meta, splits["test"])
    test_rows = splits["test"]
    prediction_data = {k: meta[k][test_rows] for k in
                       ("y", "subject", "video", "trial_idx", "window_idx", "window_id", "window_start")}
    prediction_data.update(probabilities=prediction, pred=prediction.argmax(1),
                           fold=np.full(len(test_rows), task["fold"], dtype=np.int64))
    atomic_npz(paths["predictions"], **prediction_data)
    if is_shuffled:
        donor_records.update(selection_train_rows=splits["inner_train"], refit_train_rows=splits["outer_train"],
                             selection_train_donors=np.stack(selected_donors), refit_train_donors=np.stack(refit_donors),
                             selection_train_seeds=donor_seeds["selection_train_base"] + np.arange(1, len(selected_donors) + 1),
                             refit_train_seeds=donor_seeds["refit_train_base"] + np.arange(1, len(refit_donors) + 1),
                             validation_seed=np.array(donor_seeds["validation"]), test_seed=np.array(donor_seeds["test"]))
        atomic_npz(paths["donors"], **donor_records)
    artifacts = {str(path.relative_to(root)): digest(path) for key, path in paths.items()
                 if key != "run" and (key != "donors" or is_shuffled)}
    result.update(complete=not args.smoke, production_complete=not args.smoke,
                  smoke_complete=bool(args.smoke), status="smoke_complete" if args.smoke else "complete",
                  refit_outer_train=True, selected_epoch=selected_epoch,
                  selection_epochs=len(curves["selection"]), refit_epochs=len(curves["refit"]),
                  best_validation_trial_balanced_accuracy=best_accuracy, best_validation_trial_cross_entropy=best_loss,
                  balanced_accuracy=test_accuracy, trial_cross_entropy=test_loss,
                  initial_backbone_sha256=initial_hash, refit_initial_backbone_sha256=refit_initial_hash,
                  training=training, device=str(device), torch_version=torch.__version__,
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU software smoke",
                  seconds_elapsed=time.monotonic() - started, artifact_sha256=artifacts,
                  donor_seeds=donor_seeds if is_shuffled else None,
                  **{key + "_path": str(path.relative_to(root)) for key, path in paths.items()
                     if key != "run" and (key != "donors" or is_shuffled)})
    atomic_json(paths["run"], result)
    print(json.dumps({"result": str(paths["run"]), "complete": result["complete"],
                      "smoke": args.smoke, "trial_balanced_accuracy": test_accuracy}), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-epochs", type=int, default=2, choices=(1, 2))
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    return p


if __name__ == "__main__":
    run_task(parser().parse_args())
