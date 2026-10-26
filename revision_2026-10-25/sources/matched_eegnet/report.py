"""Report matched EEGNet experiments from completed participant-held-out folds.

Ten-second predictions are averaged over three windows of the same 30-second trial.
Paired participant inference is conditional on fitted OOF models and familiar
stimuli; it is not a full-refit label-permutation test. Overlapping training sets
make the participant sign-flip calculation an approximation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

PRIMARY_SEED = 20261024
MODELS = ("eegnet", "eegnet_connectivity", "eegnet_recurrence", "eegnet_combined", "eegnet_shuffled")
LABELS = {"eegnet": "EEGNet", "eegnet_connectivity": "+ connectivity",
          "eegnet_recurrence": "+ recurrence", "eegnet_combined": "+ both",
          "eegnet_shuffled": "+ shuffled both"}
CONTRASTS = (("eegnet_connectivity", "eegnet"), ("eegnet_recurrence", "eegnet"),
             ("eegnet_combined", "eegnet"), ("eegnet_combined", "eegnet_shuffled"))
DATASET_LABELS = {"faced": "FACED", "seed7": "SEED-VII"}
VALIDATION_CRITERION = "trial balanced accuracy; lower validation loss breaks ties"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, name, length=None):
    value = np.asarray(value)
    if value.ndim != 1 or (length is not None and len(value) != length):
        raise ValueError(f"{name} must be a vector with the expected row count")
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    return value.astype(np.int64)


def probabilities(value, rows, n_classes):
    value = np.asarray(value, dtype=float)
    if value.shape != (rows, n_classes) or not np.isfinite(value).all():
        raise ValueError("Probability matrix has incorrect dimensions or nonfinite values")
    if (value < -1e-6).any() or (value > 1 + 1e-6).any() or not np.allclose(value.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Predicted class probabilities must be in [0,1] and sum to one")
    return value


def confusion(y, pred, n_classes):
    y, pred = vector(y, "y"), vector(pred, "pred")
    if len(y) != len(pred) or not len(y) or min(y.min(), pred.min()) < 0 or max(y.max(), pred.max()) >= n_classes:
        raise ValueError("Invalid labels or predictions")
    return np.bincount(y * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def score(y, pred, n_classes):
    cm = confusion(y, pred, n_classes)
    support = cm.sum(axis=1)
    if (support == 0).any():
        raise ValueError("Every declared class must appear in an evaluated set")
    return float(np.mean(np.diag(cm) / support))


def aggregate_windows(data, seconds, n_classes):
    """Average probabilities, never hard labels, across exactly 3 or 1 windows."""
    if seconds not in (10, 30):
        raise ValueError("Expected 10- or 30-second model windows")
    keys = ("y", "trial_idx", "subject", "video", "fold", "window_id", "window_start", "window_idx")
    rows = len(data["y"])
    windows = {key: vector(data[key], key, rows) for key in keys}
    windows["probabilities"] = probabilities(data["probabilities"], rows, n_classes)
    if not rows or len(np.unique(windows["window_idx"])) != rows:
        raise ValueError("Window indices must be nonempty and unique")
    n_windows = 30 // seconds
    if np.any(windows["y"] < 0) or np.any(windows["y"] >= n_classes):
        raise ValueError("Window labels are outside the declared classes")
    result = {key: [] for key in ("trial_idx", "subject", "video", "fold", "y", "probabilities")}
    for trial in np.unique(windows["trial_idx"]):
        take = np.flatnonzero(windows["trial_idx"] == trial)
        order = take[np.argsort(windows["window_id"][take])]
        if len(order) != n_windows or not np.array_equal(windows["window_id"][order], np.arange(n_windows)):
            raise ValueError("A trial has missing, duplicated, or unexpected windows")
        if not np.array_equal(windows["window_start"][order], seconds * np.arange(n_windows)):
            raise ValueError("Windows do not cover the prespecified nonoverlapping positions")
        result["trial_idx"].append(int(trial))
        for key in ("subject", "video", "fold", "y"):
            values = np.unique(windows[key][order])
            if len(values) != 1:
                raise ValueError(f"A trial's windows disagree on {key}")
            result[key].append(int(values[0]))
        result["probabilities"].append(windows["probabilities"][order].mean(axis=0))
    result = {key: np.asarray(value) for key, value in result.items()}
    if len(set(zip(result["subject"], result["video"]))) != len(result["y"]):
        raise ValueError("Multiple trial IDs identify the same participant-video pair")
    result["pred"] = result["probabilities"].argmax(axis=1)
    windows["pred"] = windows["probabilities"].argmax(axis=1)
    return result, windows


def subject_scores(data, n_classes):
    subjects = np.unique(data["subject"])
    values = np.array([score(data["y"][data["subject"] == subject],
                             data["pred"][data["subject"] == subject], n_classes) for subject in subjects])
    pooled = score(data["y"], data["pred"], n_classes)
    supports = np.array([np.bincount(data["y"][data["subject"] == subject], minlength=n_classes) for subject in subjects])
    if not np.all(supports == supports[0]):
        raise ValueError("The matched design requires identical class counts for every participant")
    if not np.isclose(pooled, values.mean(), atol=1e-12):
        raise ValueError("Pooled and mean participant balanced accuracies disagree")
    return subjects, values


def bootstrap_interval(values, n_bootstrap=2000, seed=PRIMARY_SEED):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or n_bootstrap < 2 or not np.isfinite(values).all():
        raise ValueError("Participant bootstrap needs at least two finite participant values and replicates")
    weights = np.random.default_rng(seed).multinomial(len(values), np.full(len(values), 1 / len(values)), size=n_bootstrap)
    return tuple(float(x) for x in np.quantile(weights @ values / len(values), [0.025, 0.975]))


def paired_sign_flip(differences, n_randomizations=9999, seed=PRIMARY_SEED):
    """Two-sided Monte Carlo paired sign flips; conditional approximate inference."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all() or n_randomizations < 1:
        raise ValueError("Invalid participant differences or randomization count")
    rng = np.random.default_rng(seed)
    observed = abs(float(differences.mean()))
    exceed = 0
    for start in range(0, n_randomizations, 1000):
        signs = 2 * rng.integers(0, 2, size=(min(1000, n_randomizations - start), len(differences)), dtype=np.int8) - 1
        null = np.abs(signs @ differences / len(differences))
        exceed += int(np.count_nonzero(null >= observed - 1e-12))
    return (1 + exceed) / (1 + n_randomizations)


def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if not np.isfinite(p_values).all() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("Holm correction requires all planned valid p-values")
    order = np.argsort(p_values, kind="stable")
    answer = np.empty_like(p_values)
    answer[order] = np.minimum(1, np.maximum.accumulate((len(order) - np.arange(len(order))) * p_values[order]))
    return answer


def paired_difference(first, second, n_classes, n_bootstrap, seed):
    for key in ("trial_idx", "subject", "video", "fold", "y"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Paired predictions disagree on {key}")
    subjects, a = subject_scores(first, n_classes)
    _, b = subject_scores(second, n_classes)
    difference = a - b
    low, high = bootstrap_interval(difference, n_bootstrap, seed)
    return {"gain_pp": 100 * float(difference.mean()), "ci_low_pp": 100 * low,
            "ci_high_pp": 100 * high, "n_subjects": len(subjects)}, difference


def _path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Prediction path escapes its result directory")
    return path


def validate_fold(run, data, dataset, seconds, fold, config_hash):
    if run.get("config_sha256") != config_hash:
        raise ValueError("Fold configuration hash does not match config.json")
    n_classes = len(dataset["class_names"])
    if int(run["n_classes"]) != n_classes or run.get("refit_outer_train") is not True:
        raise ValueError("Fold class count or outer-training refit declaration is incorrect")
    if int(run["selected_epoch"]) < 1 or run.get("validation_criterion") != VALIDATION_CRITERION:
        raise ValueError("Missing fixed inner-validation epoch-selection record")
    roles = {}
    for name in ("outer_train_subjects", "inner_train_subjects", "validation_subjects", "test_subjects"):
        value = vector(run[name], name)
        if not len(value) or len(value) != len(set(value)):
            raise ValueError(f"Participant role {name} is empty or duplicated")
        roles[name] = set(value.tolist())
    outer, inner, validation, test = (roles[k] for k in roles)
    if outer & test or inner & validation or inner | validation != outer:
        raise ValueError("Training/validation/test participant roles overlap or do not partition outer training")
    if len(outer | test) != int(dataset["n_subjects"]):
        raise ValueError("Outer fold does not cover the expected participant population")
    trials, windows = aggregate_windows(data, seconds, n_classes)
    if set(trials["subject"]) != test or set(trials["fold"]) != {fold}:
        raise ValueError("Saved predictions do not match the held-out participants/fold")
    videos = np.unique(trials["video"])
    if len(videos) != int(dataset["n_videos"]) or len(trials["y"]) != len(test) * len(videos):
        raise ValueError("A held-out participant is missing trial predictions")
    for video in videos:
        if len(np.unique(trials["y"][trials["video"] == video])) != 1:
            raise ValueError("Video condition labels differ across participants")
    subject_scores(trials, n_classes)
    return trials, windows, roles


def load_session_control(root, config, config_hash, n_bootstrap):
    """Verify the independent diagnostic and reconstruct its train-fold rule."""
    path = root / "session_control.json"
    if not path.exists():
        return None, {"status": "pending"}, [], {}
    from session_control import session_predictions
    record = json.loads(path.read_text())
    if record.get("complete") is not True or record.get("config_sha256") != config_hash:
        raise ValueError("Session control is incomplete or belongs to a different configuration")
    expected_metadata = config["prepared_manifest"]["seed7"]["30"]["output_sha256"]["meta.npz"]
    if record.get("metadata_sha256") != expected_metadata:
        raise ValueError("Session-control metadata hash disagrees with the prepared data")
    if record.get("source_sha256") != config["source_sha256"]["session_control.py"]:
        raise ValueError("Session-control source differs from the frozen protocol")
    if sha256(Path(__file__).with_name("session_control.py")) != record["source_sha256"]:
        raise ValueError("Current session-control implementation differs from the saved diagnostic")
    prediction_path = _path(root, record["predictions_path"])
    if sha256(prediction_path) != record.get("predictions_sha256"):
        raise ValueError("Session-control prediction checksum failed")
    with np.load(prediction_path, allow_pickle=False) as saved:
        data = {key: np.asarray(saved[key]) for key in saved.files}
    n_classes = len(config["datasets"]["seed7"]["class_names"])
    trial, _ = aggregate_windows(data, 30, n_classes)
    expected_pred, expected_fold, roles = session_predictions(
        data, n_classes, int(config["primary_seed"]), int(config["n_folds"]))
    if not np.array_equal(data["pred"], expected_pred) or not np.array_equal(data["fold"], expected_fold):
        raise ValueError("Session predictions do not reconstruct from outer-training participants")
    if record.get("fold_roles") != roles or not np.array_equal(data["probabilities"], np.eye(n_classes)[expected_pred]):
        raise ValueError("Session-control roles/probabilities disagree with its decision rule")
    subjects, scores = subject_scores(trial, n_classes)
    if len(subjects) != config["datasets"]["seed7"]["n_subjects"] or len(trial["y"]) != len(subjects) * config["datasets"]["seed7"]["n_videos"]:
        raise ValueError("Session control does not cover the full SEED-VII evaluation")
    if not np.isclose(scores.mean(), record["balanced_accuracy"], atol=1e-12):
        raise ValueError("Stored session-control accuracy is incorrect")
    low, high = bootstrap_interval(scores, n_bootstrap, int(config["primary_seed"]))
    summary = {"status": "complete", "balanced_accuracy_percent": 100 * scores.mean(),
               "ci_low_percent": 100 * low, "ci_high_percent": 100 * high,
               "n_subjects": len(subjects), "n_trials": len(trial["y"]),
               "scope": record["scope"], "metadata_sha256": expected_metadata,
               "bootstrap_scope": "Conditional on familiar sessions/clips; identical participant scores can yield degenerate intervals"}
    files = {str(p.relative_to(root)): sha256(p) for p in (path, prediction_path)}
    return trial, summary, roles, files


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def make_figures(destination, performance, contrasts, completed, config, session_control=None):
    if not completed:
        return []
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matched_eegnet_mpl_"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    primary = int(config["primary_seed"])
    contexts = list(itertools.product(config["datasets"], config["seconds"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), squeeze=False)
    for ax, (dataset, seconds) in zip(axes.flat, contexts):
        frame = performance[(performance.dataset == dataset) & (performance.seconds == seconds) & (performance.seed == primary)]
        for index, model in enumerate(config["models"]):
            row = frame[frame.model == model].iloc[0]
            if row.status == "complete":
                ax.vlines(index, row.ci_low_percent, row.ci_high_percent, color="#2166ac")
                ax.scatter(index, row.balanced_accuracy_percent, color="#2166ac", s=26)
            else:
                ax.text(index, .1, "Pending", ha="center", rotation=90, transform=ax.get_xaxis_transform(), color="#777777")
        label = "three 10-s windows; trial-averaged" if seconds == 10 else "one 30-s window"
        ax.set(title=f"{DATASET_LABELS.get(dataset, dataset)}: {label}", ylabel="Trial balanced accuracy (%)",
               xticks=np.arange(len(config["models"])), xticklabels=[LABELS[m] for m in config["models"]], xlim=(-.5, len(config["models"]) - .5))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.grid(axis="y", alpha=.2)
        if dataset == "seed7" and session_control and session_control.get("status") == "complete":
            ax.axhline(session_control["balanced_accuracy_percent"], color="#8c510a", linestyle="--", linewidth=1,
                       label="Session-only control (no EEG)")
            ax.legend(loc="best", fontsize=7, frameon=False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Matched EEGNet models in unseen participants viewing familiar videos")
    fig.text(.01, .005, "Primary training seed; 95% participant-bootstrap intervals conditional on fitted OOF models. Both window designs cover 30 s per trial.", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .96))
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"matched_performance.{suffix}", dpi=220)
    plt.close(fig)
    primary_rows = contrasts[contrasts.seed == primary]
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = []
    for index, row in enumerate(primary_rows.itertuples()):
        label = f"{DATASET_LABELS.get(row.dataset, row.dataset)} | {'3 × 10 s' if row.seconds == 10 else '30 s'} | {LABELS[row.model_a]} vs {LABELS[row.model_b]}"
        labels.append(label)
        if row.status == "complete":
            ax.hlines(index, row.ci_low_pp, row.ci_high_pp, color="#2166ac")
            ax.scatter(row.gain_pp, index, color="#2166ac", s=25)
        else:
            ax.text(.98, index, "Pending", transform=ax.get_yaxis_transform(), ha="right", color="#777777")
    ax.set(yticks=np.arange(len(labels)), yticklabels=labels, xlabel="Paired difference in trial balanced accuracy (percentage points)",
           title="Prespecified augmentation and shuffled-input comparisons")
    ax.invert_yaxis(); ax.axvline(0, color="#666666", linewidth=.8); ax.grid(axis="x", alpha=.2)
    fig.text(.01, .005, "95% paired participant-bootstrap intervals; primary seed. Holm-adjusted, approximate paired sign-flip tests are reported in the comparison table.", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"matched_increment_forest.{suffix}", dpi=220)
    plt.close(fig)
    confusion_file = destination / "matched_confusion_matrices.pdf"
    with PdfPages(confusion_file) as pages:
        for dataset, seconds in contexts:
            classes = config["datasets"][dataset]["class_names"]
            fig, axes = plt.subplots(1, len(config["models"]), figsize=(20, 4.5), squeeze=False)
            for ax, model in zip(axes[0], config["models"]):
                trial = completed.get((dataset, seconds, model, primary))
                ax.set_title(LABELS[model])
                if trial is None:
                    ax.axis("off"); ax.text(.5, .5, "Pending", ha="center", transform=ax.transAxes); continue
                cm = confusion(trial["y"], trial["pred"], len(classes))
                values = 100 * cm / cm.sum(axis=1, keepdims=True)
                ax.imshow(values, vmin=0, vmax=100, cmap="Blues")
                ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes,
                       yticklabels=classes, xlabel="Predicted condition", ylabel="Assigned video condition")
                ax.tick_params(axis="x", labelrotation=70, labelsize=7); ax.tick_params(axis="y", labelsize=7)
                for a, b in itertools.product(range(len(classes)), repeat=2):
                    ax.text(b, a, f"{values[a,b]:.0f}", ha="center", va="center", fontsize=6, color="white" if values[a,b] > 50 else "#222222")
            fig.suptitle(f"{DATASET_LABELS.get(dataset, dataset)}: {'three 10-s windows, trial-averaged' if seconds == 10 else 'one 30-s window'}; within-condition percentages")
            fig.tight_layout(rect=(0, 0, 1, .93)); pages.savefig(fig); plt.close(fig)
    return ["matched_performance.pdf", "matched_performance.png", "matched_increment_forest.pdf", "matched_increment_forest.png", confusion_file.name]


def build_report(root, n_bootstrap=2000, n_randomizations=9999, figures=True):
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    primary = int(config.get("primary_seed", PRIMARY_SEED))
    seeds = [int(x) for x in config.get("seeds", [primary, 11, 29])]
    datasets, durations, models = config["datasets"], [int(x) for x in config["seconds"]], config["models"]
    n_folds = int(config.get("n_folds", 5))
    if set(datasets) != {"faced", "seed7"} or set(durations) != {10, 30} or set(models) != set(MODELS):
        raise ValueError("The fixed comparison family requires both datasets, both window designs, and all five EEGNet variants")
    if len(seeds) != len(set(seeds)) or primary not in seeds or n_folds < 2:
        raise ValueError("Invalid seed/fold specification")
    config = dict(config, primary_seed=primary, seeds=seeds, seconds=durations)
    runs, files, role_reference = {}, {}, {}
    for path in sorted((root / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        if run.get("complete") is not True:
            continue
        key = (str(run["dataset"]), int(run["seconds"]), str(run["model"]), int(run["seed"]), int(run["fold"]))
        dataset, seconds, model, seed, fold = key
        if dataset not in datasets or seconds not in durations or model not in models or seed not in seeds or fold not in range(n_folds):
            raise ValueError(f"Completed run is outside the fixed analysis configuration: {path.name}")
        if key in runs:
            raise ValueError(f"Duplicate completed fold: {key}")
        prediction_path = _path(root, run["predictions_path"])
        expected_artifact = run.get("artifact_sha256", {}).get(run["predictions_path"])
        if expected_artifact is None or sha256(prediction_path) != expected_artifact:
            raise ValueError("Completed fold prediction checksum is missing or does not match")
        with np.load(prediction_path, allow_pickle=False) as handle:
            data = {k: np.asarray(handle[k]) for k in handle.files}
        trials, windows, roles = validate_fold(run, data, datasets[dataset], seconds, fold, config_hash)
        role_key = (dataset, fold)
        if role_key in role_reference and role_reference[role_key] != roles:
            raise ValueError("Models, window designs, or training seeds use different participant splits")
        role_reference[role_key] = roles
        runs[key] = (trials, windows, run)
        files[str(path.relative_to(root))] = sha256(path)
        files[str(prediction_path.relative_to(root))] = sha256(prediction_path)
    session_trial, session_summary, session_roles, session_files = load_session_control(root, config, config_hash, n_bootstrap)
    files.update(session_files)
    for control_role in session_roles:
        reference = role_reference.get(("seed7", control_role["fold"]))
        if reference is not None and reference != {key: set(value) for key, value in control_role.items() if key != "fold"}:
            raise ValueError("EEG and session-control participant roles differ")
    performance, completed, windows_table, confusion_rows, fold_rows = [], {}, [], [], []
    destination = root / "report"
    (destination / "trial_predictions").mkdir(parents=True, exist_ok=True)
    for dataset, seconds, model, seed in itertools.product(datasets, durations, models, seeds):
        context = (dataset, seconds, model, seed)
        available = [runs[context + (fold,)] for fold in range(n_folds) if context + (fold,) in runs]
        row = {"dataset": dataset, "seconds": seconds, "model": model, "seed": seed,
               "role": "primary" if seed == primary else "sensitivity", "status": "pending",
               "completed_folds": len(available), "expected_folds": n_folds,
               "balanced_accuracy_percent": None, "ci_low_percent": None, "ci_high_percent": None,
               "window_pooled_balanced_accuracy_percent": None, "n_subjects": None, "n_trials": None}
        if len(available) == n_folds:
            data = {key: np.concatenate([item[1][key] for item in available]) for key in available[0][1]}
            trial, window = aggregate_windows(data, seconds, len(datasets[dataset]["class_names"]))
            if len(np.unique(trial["subject"])) != int(datasets[dataset]["n_subjects"]) or len(trial["y"]) != int(datasets[dataset]["n_subjects"]) * int(datasets[dataset]["n_videos"]):
                raise ValueError("Complete folds do not cover every expected participant and trial")
            for subject in np.unique(trial["subject"]):
                if len(np.unique(trial["fold"][trial["subject"] == subject])) != 1:
                    raise ValueError("A participant appears in multiple test folds")
            n_classes = len(datasets[dataset]["class_names"])
            subjects, values = subject_scores(trial, n_classes)
            low, high = bootstrap_interval(values, n_bootstrap, primary)
            row.update(status="complete", balanced_accuracy_percent=100 * values.mean(), ci_low_percent=100 * low,
                       ci_high_percent=100 * high, n_subjects=len(subjects), n_trials=len(trial["y"]),
                       window_pooled_balanced_accuracy_percent=100 * score(window["y"], window["pred"], n_classes))
            completed[context] = trial
            np.savez_compressed(destination / "trial_predictions" / f"{dataset}_{seconds}s_{model}_seed{seed}.npz", **trial)
            for window_id in np.unique(window["window_id"]):
                take = window["window_id"] == window_id
                windows_table.append({**dict(zip(("dataset", "seconds", "model", "seed"), context)), "window_id": int(window_id),
                                      "window_start": int(window_id * seconds), "balanced_accuracy_percent": 100 * score(window["y"][take], window["pred"][take], n_classes),
                                      "n_windows": int(take.sum()), "scope": "descriptive; repeated windows are not independent participants"})
            cm = confusion(trial["y"], trial["pred"], n_classes)
            for a, b in itertools.product(range(n_classes), repeat=2):
                confusion_rows.append({**dict(zip(("dataset", "seconds", "model", "seed"), context)), "true_class": a, "predicted_class": b,
                                       "true_label": datasets[dataset]["class_names"][a], "predicted_label": datasets[dataset]["class_names"][b],
                                       "n_trials": int(cm[a,b]), "row_percent": 100 * cm[a,b] / cm[a].sum()})
            for fold_trial, _, run in available:
                fold_rows.append({**dict(zip(("dataset", "seconds", "model", "seed"), context)), "fold": int(run["fold"]),
                                  "selected_epoch": int(run["selected_epoch"]), "n_test_subjects": len(np.unique(fold_trial["subject"])),
                                  "balanced_accuracy_percent": 100 * score(fold_trial["y"], fold_trial["pred"], n_classes)})
        performance.append(row)
    contrasts = []
    for dataset, seconds, seed, (a, b) in itertools.product(datasets, durations, seeds, CONTRASTS):
        row = {"dataset": dataset, "seconds": seconds, "seed": seed, "model_a": a, "model_b": b,
               "status": "pending", "role": "primary" if seed == primary else "sensitivity",
               "gain_pp": None, "ci_low_pp": None, "ci_high_pp": None, "n_subjects": None,
               "p_paired_sign_flip": None, "p_holm": None}
        first, second = completed.get((dataset, seconds, a, seed)), completed.get((dataset, seconds, b, seed))
        if first is not None and second is not None:
            summary, differences = paired_difference(first, second, len(datasets[dataset]["class_names"]), n_bootstrap, primary)
            row.update(summary, status="complete")
            if seed == primary:
                row["p_paired_sign_flip"] = paired_sign_flip(differences, n_randomizations, primary)
        contrasts.append(row)
    primary_rows = [row for row in contrasts if row["seed"] == primary]
    holm_complete = len(primary_rows) == 16 and all(row["p_paired_sign_flip"] is not None for row in primary_rows)
    if holm_complete:
        for row, adjusted in zip(primary_rows, holm([row["p_paired_sign_flip"] for row in primary_rows])):
            row["p_holm"] = float(adjusted)
    duration_rows = []
    for dataset, model, seed in itertools.product(datasets, models, seeds):
        row = {"dataset": dataset, "model": model, "seed": seed, "status": "pending",
               "contrast": "three 10-s windows averaged minus one 30-s window; both cover 30 s",
               "gain_pp": None, "ci_low_pp": None, "ci_high_pp": None, "n_subjects": None}
        first, second = completed.get((dataset, 10, model, seed)), completed.get((dataset, 30, model, seed))
        if first is not None and second is not None:
            summary, _ = paired_difference(first, second, len(datasets[dataset]["class_names"]), n_bootstrap, primary)
            row.update(summary, status="complete")
        duration_rows.append(row)
    session_rows = []
    for seconds, model, seed in itertools.product(durations, models, seeds):
        row = {"dataset": "seed7", "seconds": seconds, "model": model, "seed": seed,
               "control": "session_only", "status": "pending", "gain_pp": None,
               "ci_low_pp": None, "ci_high_pp": None, "n_subjects": None,
               "scope": "descriptive paired participant interval; no additional hypothesis test"}
        trial = completed.get(("seed7", seconds, model, seed))
        if trial is not None and session_trial is not None:
            summary, _ = paired_difference(trial, session_trial, len(datasets["seed7"]["class_names"]), n_bootstrap, primary)
            row.update(summary, status="complete")
        session_rows.append(row)
    tables = {"performance.csv": pd.DataFrame(performance), "window_performance.csv": pd.DataFrame(windows_table),
              "paired_augmentation_comparisons.csv": pd.DataFrame(contrasts), "duration_comparisons.csv": pd.DataFrame(duration_rows),
              "session_control_comparisons.csv": pd.DataFrame(session_rows),
              "confusion_matrices.csv": pd.DataFrame(confusion_rows), "fold_performance.csv": pd.DataFrame(fold_rows)}
    for name, table in tables.items():
        table.to_csv(destination / name, index=False)
    figure_files = make_figures(destination, tables["performance.csv"], tables["paired_augmentation_comparisons.csv"], completed, config, session_summary) if figures else []
    result = {"protocol": "Matched EEGNet augmentation comparison in unseen participants viewing familiar stimuli",
              "primary_seed": primary, "seeds": seeds, "completed_fold_runs": len(runs), "expected_fold_runs": len(datasets) * len(durations) * len(models) * len(seeds) * n_folds,
              "completed_model_runs": len(completed), "expected_primary_comparisons": 16,
              "holm_status": "complete" if holm_complete else "pending",
              "bootstrap_replicates": n_bootstrap, "paired_sign_flip_replicates": n_randomizations,
              "inferential_scope": "Two-sided participant sign flips assume exchangeability of paired model scores; conditional on fitted OOF models. Overlapping training sets make this an approximation; it excludes training-instability uncertainty. These are not full-refit label-permutation tests and do not establish performance above a stimulus-structured null.",
              "generalization_scope": "New participants, familiar videos and sessions; models trained within each dataset. Not unseen-stimulus generalization or transfer of FACED-trained weights to SEED-VII.",
              "temporal_scope": "Three nonoverlapping 10-s windows are probability-averaged per trial and compared with one 30-s window. Both designs use 30 s; individual-window accuracies are descriptive.",
              "multiplicity_scope": "One Holm family of 16 primary-seed comparisons: three augmentations versus EEGNet and combined versus shuffled-combined, across two datasets and two window designs. Holm values withheld until every comparison is complete. Sensitivity seeds and duration comparisons are descriptive.",
              "ci_scope": "Paired participant bootstrap retaining all familiar videos/windows; conditional on fitted models and stimulus set. Pointwise 95% intervals; not simultaneous intervals.",
              "feature_scope": "Augmentations are engineered descriptions of the same EEG window; improvement does not establish independent physiological information or causal relevance.",
              "session_control": session_summary, "session_control_comparisons": session_rows,
              "session_scope": "The SEED-VII session-only diagnostic uses no EEG and fits the class-balanced session decision from outer-training participants. The same trial predictions apply to both window designs. Merely exceeding nominal chance or this control does not establish session-independent emotion information; control comparisons have descriptive intervals only.",
              "performance": performance, "comparisons": contrasts, "duration_comparisons": duration_rows,
              "config_sha256": config_hash, "config_file_sha256": sha256(root / "config.json"),
              "input_file_sha256": files, "report_script_sha256": sha256(__file__), "figures": figure_files}
    (destination / "report.json").write_text(json.dumps(_safe_json(result), indent=2, allow_nan=False) + "\n")
    lines = ["Matched EEGNet benchmark", "", result["protocol"], "", result["generalization_scope"], "", result["temporal_scope"], "",
             "Epoch selection uses participant-disjoint inner validation and trial-level balanced accuracy, with validation loss breaking ties.",
             "Each outer model is then refitted from fresh weights on all outer training participants for the selected epoch count.",
             "Feature normalization uses training participants only. Connectivity and recurrence use the same input window as EEGNet.",
             "The combined and shuffled-combined models have identical input dimensions and capacity; their auxiliary features are jointly shuffled within participant.",
             "", result["ci_scope"], "", result["inferential_scope"], "", result["multiplicity_scope"], "", result["feature_scope"], "", result["session_scope"], "",
             "Missing folds remain pending: partial OOF predictions do not generate model scores or comparative inference.",
             "Positive paired differences favor the first named model; significant negative differences do not support an improvement.",
             "The report does not modify manuscript files."]
    (destination / "README.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report_directory": str(destination), "completed_fold_runs": len(runs), "completed_model_runs": len(completed), "holm_status": result["holm_status"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--randomizations", type=int, default=9999)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    build_report(args.out_dir, args.bootstrap, args.randomizations, not args.no_figures)


if __name__ == "__main__":
    main()
