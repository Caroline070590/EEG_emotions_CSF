"""Report completed participant-held-out EEG prediction runs without refitting models.

All uncertainty intervals resample participants, retaining their full set of videos.
They condition on the fitted out-of-fold models, the sampled familiar videos, and
(for feature importance) the saved feature-permutation repetitions. Null inference
requires every planned permutation at the primary seed. An incomplete null is never
used for a p-value or an above-null claim. Feature importance measures model reliance,
not causal relevance or a unique physiological contribution.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

DEFAULT_MODELS = ("hybrid_gnn", "sfgnn", "lda_full")
MODEL_LABELS = {"hybrid_gnn": "Hybrid GNN", "sfgnn": "Summary-first GNN",
                "lda_full": "Shrinkage LDA"}
DEFAULT_SEED = 20261024
FAMILY_DISPLAY = ("power_all", "rqa_all", "wpli_all", "oaec_all", "rho_signed", "joint_recurrence")
GROUP_LABELS = {"power_all": "Band power", "rqa_all": "Recurrence measures",
                "wpli_all": "wPLI", "oaec_all": "Orthogonalized AEC",
                "rho_signed": "Signed Spearman", "joint_recurrence": "Joint recurrence"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(name, value, length=None):
    result = np.asarray(value)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a vector with the expected row count")
    return result


def confusion_counts(y, pred, n_classes):
    y = _vector("y", y).astype(np.int64)
    pred = _vector("pred", pred, len(y)).astype(np.int64)
    if not len(y) or min(y.min(), pred.min()) < 0 or max(y.max(), pred.max()) >= n_classes:
        raise ValueError("Labels or predictions fall outside the declared classes")
    return np.bincount(y * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def balanced_accuracy(y, pred, n_classes):
    cm = confusion_counts(y, pred, n_classes)
    support = cm.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Balanced accuracy requires every declared class in the evaluation set")
    return float(np.mean(np.diag(cm) / support))


def participant_components(y, pred, subject, n_classes):
    """Return subject IDs and per-subject/per-class correct and total counts."""
    y = _vector("y", y).astype(np.int64)
    pred = _vector("pred", pred, len(y)).astype(np.int64)
    subject = _vector("subject", subject, len(y))
    confusion_counts(y, pred, n_classes)
    subjects, inverse = np.unique(subject, return_inverse=True)
    keys = inverse * n_classes + y
    shape = (len(subjects), n_classes)
    support = np.bincount(keys, minlength=np.prod(shape)).reshape(shape).astype(float)
    correct = np.bincount(keys, weights=(y == pred), minlength=np.prod(shape)).reshape(shape)
    return subjects, correct, support


def bootstrap_weights(n_subjects, n_bootstrap=2000, seed=DEFAULT_SEED):
    if n_subjects < 2 or n_bootstrap < 2:
        raise ValueError("At least two participants and bootstrap replicates are required")
    return np.random.default_rng(seed).multinomial(
        n_subjects, np.full(n_subjects, 1.0 / n_subjects), size=n_bootstrap)


def component_score(correct, support):
    den = support.sum(axis=0)
    if np.any(den <= 0):
        raise ValueError("A declared class has no evaluation examples")
    return float(np.mean(correct.sum(axis=0) / den))


def bootstrap_scores(correct, support, weights):
    denominator = weights @ support
    if np.any(denominator <= 0):
        raise ValueError("A participant bootstrap omitted an entire class")
    return np.mean((weights @ correct) / denominator, axis=1)


def participant_bootstrap(y, pred, subject, n_classes=9, n_bootstrap=2000, seed=DEFAULT_SEED):
    subjects, correct, support = participant_components(y, pred, subject, n_classes)
    scores = bootstrap_scores(correct, support, bootstrap_weights(len(subjects), n_bootstrap, seed))
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return {"balanced_accuracy": component_score(correct, support), "ci_low": float(lo),
            "ci_high": float(hi), "n_subjects": len(subjects), "n_trials": len(y)}


def paired_model_difference(first, second, n_classes=9, n_bootstrap=2000, seed=DEFAULT_SEED):
    """Paired participant-bootstrap difference; no null-based superiority claim."""
    for key in ("idx", "subject", "video", "y", "fold"):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f"Paired models disagree on {key}; align the same OOF trials first")
    subjects, ca, support = participant_components(first["y"], first["pred"], first["subject"], n_classes)
    _, cb, other_support = participant_components(second["y"], second["pred"], second["subject"], n_classes)
    if not np.array_equal(support, other_support):
        raise ValueError("Paired models have different class support")
    weights = bootstrap_weights(len(subjects), n_bootstrap, seed)
    distribution = bootstrap_scores(ca - cb, support, weights)
    lo, hi = np.quantile(distribution, [0.025, 0.975])
    return {"difference_pp": 100 * component_score(ca - cb, support),
            "ci_low_pp": 100 * float(lo), "ci_high_pp": 100 * float(hi),
            "n_subjects": len(subjects), "n_trials": len(first["y"])}


def permutation_summary(observed, scores, expected=99):
    """scores maps integer permutation IDs to BA; use no incomplete null inference."""
    required = set(range(1, expected + 1))
    available = required.intersection(scores)
    missing = sorted(required - available)
    result = {"null_completed": len(available), "null_expected": expected,
              "null_status": "pending" if missing else "complete",
              "missing_permutations": missing, "null_mean_percent": None,
              "null_95th_percentile_percent": None, "gain_pp": None,
              "relative_gain_percent": None, "p_raw": None, "p_holm": None}
    if missing or observed is None:
        if observed is None:
            result["null_status"] = "observed_pending" if not missing else "pending"
        return result
    values = np.asarray([scores[i] for i in sorted(required)], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Permutation balanced accuracies must be finite and in [0,1]")
    mean = float(values.mean())
    result.update(null_mean_percent=100 * mean,
                  null_95th_percentile_percent=100 * float(np.quantile(values, 0.95)),
                  gain_pp=100 * (observed - mean),
                  relative_gain_percent=100 * (observed - mean) / mean if mean > 0 else None,
                  p_raw=float((1 + np.count_nonzero(values >= observed)) / (expected + 1)))
    return result


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm correction requires all planned finite p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]))
    output = np.empty_like(values)
    output[order] = adjusted
    return output


def grouped_importance(oof, predictions, group_names, group_kinds, n_classes=9,
                       n_bootstrap=2000, seed=DEFAULT_SEED):
    """OOF permutation importance, averaging repeats before participant resampling.

    Return group, fold, participant and rank-stability tables. Positive importance
    means lower balanced accuracy after replacing that group. Negative values are
    preserved. Family and atomic groups overlap, so their scores must not be summed.
    """
    predictions = np.asarray(predictions)
    group_names, group_kinds = list(group_names), list(group_kinds)
    expected = (len(group_names), predictions.shape[1] if predictions.ndim == 3 else -1, len(oof["y"]))
    if predictions.ndim != 3 or predictions.shape != expected or predictions.shape[1] < 1:
        raise ValueError("Importance predictions must have shape (groups, repeats, OOF rows)")
    if len(group_names) != len(set(group_names)) or len(group_kinds) != len(group_names):
        raise ValueError("Importance group names and kinds are inconsistent")
    if not np.isin(group_kinds, ["atomic", "family"]).all():
        raise ValueError("Expected atomic or family group kinds")
    if predictions.min() < 0 or predictions.max() >= n_classes:
        raise ValueError("Importance predictions contain invalid classes")
    subjects, base_correct, support = participant_components(oof["y"], oof["pred"], oof["subject"], n_classes)
    if np.any(support == 0):
        raise ValueError("Participant importance requires every participant to have every class")
    weights = bootstrap_weights(len(subjects), n_bootstrap, seed)
    inverse = np.searchsorted(subjects, oof["subject"])
    keys = inverse * n_classes + oof["y"]
    n_repeats = predictions.shape[1]
    subject_folds = np.array([np.unique(oof["fold"][oof["subject"] == s]).item() for s in subjects])
    rows, fold_rows, participant_rows = [], [], []
    for g, (name, kind) in enumerate(zip(group_names, group_kinds)):
        repeat_correct = np.stack([
            np.bincount(keys, weights=(predictions[g, rep] == oof["y"]),
                        minlength=len(subjects) * n_classes).reshape(support.shape)
            for rep in range(n_repeats)])
        mean_correct = repeat_correct.mean(axis=0)
        delta = base_correct - mean_correct
        samples = bootstrap_scores(delta, support, weights)
        ci = np.quantile(samples, [0.025, 0.975])
        repeat_scores = np.array([component_score(base_correct - v, support) for v in repeat_correct])
        rows.append({"group": str(name), "kind": str(kind),
                     "mean_drop_pp": 100 * component_score(delta, support),
                     "ci_low_pp": 100 * float(ci[0]), "ci_high_pp": 100 * float(ci[1]),
                     "shuffle_repeat_sd_pp": 100 * float(repeat_scores.std(ddof=1)) if n_repeats > 1 else 0,
                     "n_subjects": len(subjects), "n_repeats": n_repeats})
        participant_drop = (delta / support).mean(axis=1)
        for j, subject in enumerate(subjects):
            participant_rows.append({"group": str(name), "kind": str(kind), "subject": int(subject),
                                     "fold": int(subject_folds[j]), "mean_drop_pp": 100 * float(participant_drop[j])})
        for fold in np.unique(subject_folds):
            take = subject_folds == fold
            fold_rows.append({"group": str(name), "kind": str(kind), "fold": int(fold),
                              "mean_drop_pp": 100 * component_score(delta[take], support[take]),
                              "n_subjects": int(take.sum())})
    group_table, fold_table = pd.DataFrame(rows), pd.DataFrame(fold_rows)
    fold_table["rank_within_kind"] = fold_table.groupby(["fold", "kind"])["mean_drop_pp"].rank(
        ascending=False, method="average")
    rank_summary = fold_table.groupby(["group", "kind"])["rank_within_kind"].agg(
        mean_fold_rank="mean", sd_fold_rank="std", min_fold_rank="min", max_fold_rank="max").reset_index()
    tie_inclusive_rank = fold_table.groupby(["fold", "kind"])["mean_drop_pp"].rank(ascending=False, method="min")
    top = fold_table.assign(top_rank=tie_inclusive_rank <= 1,
                            top_three=tie_inclusive_rank <= 3).groupby(["group", "kind"])[["top_rank", "top_three"]].sum().reset_index()
    group_table = group_table.merge(rank_summary, on=["group", "kind"]).merge(top, on=["group", "kind"])
    rank_pairs = []
    for kind, frame in fold_table.groupby("kind"):
        pivot = frame.pivot(index="group", columns="fold", values="rank_within_kind")
        for first, second in itertools.combinations(pivot.columns, 2):
            x, y = pivot[first].to_numpy(), pivot[second].to_numpy()
            rho = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
            rank_pairs.append({"kind": kind, "fold_a": int(first), "fold_b": int(second),
                               "spearman_rank_correlation": rho, "n_groups": len(x)})
    return group_table, fold_table, pd.DataFrame(participant_rows), pd.DataFrame(rank_pairs)


def _within_root(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Result path is outside its analysis directory: {relative}")
    return path


def load_oof(root, run, n_classes, n_folds, n_subjects=None, n_videos=None):
    path = _within_root(root, run["predictions_path"])
    with np.load(path, allow_pickle=False) as data:
        oof = {k: np.asarray(data[k]) for k in ("y", "pred", "subject", "video", "idx", "fold")}
        if int(run["perm"]) == 0:
            probabilities = np.asarray(data["probabilities"])
            if probabilities.shape != (len(oof["y"]), n_classes) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid probability matrix: {path}")
            if np.any(probabilities < -1e-6) or np.any(probabilities > 1 + 1e-6) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
                raise ValueError(f"Probabilities do not sum to one: {path}")
    n = len(oof["y"])
    for key, value in oof.items():
        _vector(key, value, n)
    if len(np.unique(oof["idx"])) != n or len(set(zip(oof["subject"], oof["video"]))) != n:
        raise ValueError(f"Duplicate OOF trials in {path}")
    order = np.argsort(oof["idx"])
    oof = {key: value[order] for key, value in oof.items()}
    subjects, videos = np.unique(oof["subject"]), np.unique(oof["video"])
    if len(np.unique(oof["fold"])) != n_folds or n != len(subjects) * len(videos):
        raise ValueError(f"Incomplete participant-fold coverage: {path}")
    if n_subjects is not None and len(subjects) != n_subjects:
        raise ValueError(f"Incorrect participant coverage: {path}")
    if n_videos is not None and len(videos) != n_videos:
        raise ValueError(f"Incorrect video coverage: {path}")
    if any(len(np.unique(oof["fold"][oof["subject"] == s])) != 1 for s in subjects):
        raise ValueError(f"A participant appears in multiple held-out folds: {path}")
    if any(len(np.unique(oof["y"][oof["video"] == v])) != 1 for v in videos):
        raise ValueError(f"Video-label assignments differ across participants: {path}")
    value = balanced_accuracy(oof["y"], oof["pred"], n_classes)
    if not np.isclose(value, run["balanced_accuracy"], atol=1e-7, rtol=0):
        raise ValueError(f"Stored and reconstructed balanced accuracies disagree: {path}")
    for item in run.get("fold_metrics", []):
        mask = oof["fold"] == int(item["fold"])
        if not mask.any():
            raise ValueError(f"Stored fold has no predictions: {path}")
        if set(item.get("train_subjects", [])).intersection(item.get("test_subjects", [])):
            raise ValueError(f"Training and test participants overlap: {path}")
        if "test_subjects" in item and set(item["test_subjects"]) != set(oof["subject"][mask]):
            raise ValueError(f"Stored test participants disagree with predictions: {path}")
        if "balanced_accuracy" in item and not np.isclose(
                balanced_accuracy(oof["y"][mask], oof["pred"][mask], n_classes),
                item["balanced_accuracy"], atol=1e-7, rtol=0):
            raise ValueError(f"Stored fold accuracy disagrees with predictions: {path}")
    return oof, order, path


def _plot_setup():
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="participant_report_mpl_")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    return plt


def _save_figure(fig, prefix):
    for suffix in ("pdf", "png"):
        fig.savefig(str(prefix) + "." + suffix, bbox_inches="tight", facecolor="white")


def _draw_confusion(ax, oof, class_names, title):
    cm = confusion_counts(oof["y"], oof["pred"], len(class_names))
    normalized = 100 * cm / cm.sum(axis=1, keepdims=True)
    image = ax.imshow(normalized, vmin=0, vmax=100, cmap="Blues")
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, xlabel="Predicted condition",
           ylabel="Assigned video condition", title=title)
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{normalized[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if normalized[i, j] > 50 else "#202020")
    return image


def make_figures(destination, performance, observed, importance, class_names, primary_seed, model_names):
    plt = _plot_setup()
    primary = performance[performance.seed == primary_seed]
    complete = primary[primary.status == "complete"]
    if complete.empty:
        return []
    focus = "hybrid_gnn" if "hybrid_gnn" in model_names else model_names[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [0.9, 1.35, 1.25]})
    palette = {model: color for model, color in zip(model_names, ["#2166ac", "#7b3294", "#1b7837"])}
    for j, row in enumerate(primary.itertuples()):
        if row.status == "complete":
            color = palette.get(row.model, "#444444")
            axes[0].vlines(j, row.ci_low_percent, row.ci_high_percent, color=color)
            axes[0].scatter(j, row.balanced_accuracy_percent, color=color, zorder=3)
            if row.null_status == "complete":
                axes[0].scatter(j, row.null_mean_percent, marker="x", color="#707070", zorder=3)
                caption = (f"null complete\nHolm p={row.p_holm:.3f}" if pd.notna(row.p_holm)
                           else f"null complete\nraw p={row.p_raw:.3f}")
            else:
                caption = f"null pending\n{row.null_completed}/{row.null_expected}"
        else:
            caption = "observed\npending"
            axes[0].text(j, 0.4, "Pending", ha="center", va="center", rotation=90,
                         transform=axes[0].get_xaxis_transform(), fontsize=8, color="#888888")
        axes[0].text(j, -0.23, caption, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", fontsize=7, color="#555555")
    axes[0].set(xticks=np.arange(len(primary)), xticklabels=[{"hybrid_gnn": "Hybrid\nGNN", "sfgnn": "Summary-first\nGNN", "lda_full": "Shrinkage\nLDA"}.get(m, m) for m in primary.model],
                ylabel="Balanced accuracy (%)", title="a  Prediction across participants")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_xlim(-0.5, len(primary) - 0.5)
    axes[0].set_ylim(0, max(25, float(complete.ci_high_percent.max()) * 1.15))
    axes[0].grid(axis="y", alpha=0.2)
    key = (focus, primary_seed)
    if key in observed:
        cm = _draw_confusion(axes[1], observed[key], class_names, "b  " + MODEL_LABELS.get(focus, focus))
        fig.colorbar(cm, ax=axes[1], fraction=0.045, pad=0.025, label="Within-condition trials (%)")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, f"{MODEL_LABELS.get(focus, focus)}\nobserved run pending", ha="center", transform=axes[1].transAxes)
    table = importance.get(key)
    if table is not None and not table.empty:
        family = table[table.group.isin(FAMILY_DISPLAY)].copy()
        if family.empty:
            family = table[table.kind == "family"].copy()
        family = family.sort_values("mean_drop_pp", ascending=True)
        axes[2].hlines(np.arange(len(family)), family.ci_low_pp, family.ci_high_pp, color=palette.get(focus, "#2166ac"))
        axes[2].scatter(family.mean_drop_pp, np.arange(len(family)), color=palette.get(focus, "#2166ac"), s=22, zorder=3)
        axes[2].set(yticks=np.arange(len(family)), yticklabels=[GROUP_LABELS.get(x, x) for x in family.group],
                    xlabel="Balanced-accuracy drop (percentage points)", title="c  Feature-group reliance")
        axes[2].axvline(0, color="#666666", linewidth=0.8)
        axes[2].grid(axis="x", alpha=0.2)
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Feature-permutation results pending", ha="center", transform=axes[2].transAxes)
    fig.suptitle("Prediction of familiar video conditions in unseen participants", fontsize=12)
    fig.text(0.02, -0.03, "Primary seed; 95% participant-bootstrap intervals. Feature reliance is conditional on fitted models and is not causal.\n"
             "Nulls shuffle video labels jointly across participants and fully refit models; inference is withheld until all planned permutations complete.", fontsize=8)
    fig.tight_layout(pad=1.6)
    _save_figure(fig, destination / "participant_prediction_overview")
    plt.close(fig)
    figure_names = ["participant_prediction_overview.pdf", "participant_prediction_overview.png"]
    available = [(m, observed[(m, primary_seed)]) for m in model_names if (m, primary_seed) in observed]
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.8), squeeze=False)
    for ax, (model, oof) in zip(axes[0], available):
        image = _draw_confusion(ax, oof, class_names, MODEL_LABELS.get(model, model))
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025, label="Within-condition trials (%)")
    fig.suptitle("Unseen participants; familiar video conditions", fontsize=12)
    fig.tight_layout(pad=1.6)
    _save_figure(fig, destination / "participant_confusion_matrices")
    plt.close(fig)
    figure_names += ["participant_confusion_matrices.pdf", "participant_confusion_matrices.png"]
    if any((model, primary_seed) in importance for model in model_names):
        fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.6), squeeze=False, sharey=True, sharex=True)
        for ax, model in zip(axes[0], model_names):
            source = importance.get((model, primary_seed))
            ax.set_title(MODEL_LABELS.get(model, model))
            ax.set_xlabel("Balanced-accuracy drop (percentage points)")
            ax.axvline(0, color="#666666", linewidth=0.8)
            ax.grid(axis="x", alpha=0.2)
            ax.set_yticks(np.arange(len(FAMILY_DISPLAY)))
            ax.set_yticklabels([GROUP_LABELS.get(x, x) for x in FAMILY_DISPLAY])
            if source is None:
                ax.text(0.5, 0.5, "Feature permutations pending", ha="center", transform=ax.transAxes)
                continue
            family = source.set_index("group").reindex(FAMILY_DISPLAY).dropna(subset=["mean_drop_pp"])
            positions = [FAMILY_DISPLAY.index(x) for x in family.index]
            color = palette.get(model, "#444444")
            ax.hlines(positions, family.ci_low_pp, family.ci_high_pp, color=color)
            ax.scatter(family.mean_drop_pp, positions, color=color, s=24, zorder=3)
        axes[0, 0].invert_yaxis()
        fig.suptitle("Feature-group reliance in held-out participants", fontsize=12)
        fig.text(0.02, -0.025, "Same six nonoverlapping input blocks; 95% participant-bootstrap intervals. Scores measure model reliance, not causal contribution.", fontsize=8)
        fig.tight_layout(pad=1.6)
        _save_figure(fig, destination / "participant_feature_importance")
        plt.close(fig)
        figure_names += ["participant_feature_importance.pdf", "participant_feature_importance.png"]
    return figure_names


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def build_report(root, expected_permutations=99, n_bootstrap=2000):
    root = Path(root).resolve()
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    primary_seed = int(config.get("primary_seed", DEFAULT_SEED))
    model_names = list(config.get("model_names", DEFAULT_MODELS))
    if len(set(model_names)) != len(model_names) or not model_names:
        raise ValueError("Model names must be nonempty and unique")
    expected = int(config.get("expected_permutations", expected_permutations))
    if expected != expected_permutations or expected < 1:
        raise ValueError("CLI permutation count must agree with the fixed analysis configuration")
    n_folds = int(config.get("n_folds", 5))
    n_classes = int(config.get("n_classes", 9))
    manifest = config.get("data_manifest", {})
    n_subjects = config.get("n_subjects", manifest.get("n_participants"))
    n_videos = config.get("n_videos", manifest.get("n_videos"))
    class_names = config.get("class_names", [f"Class {i}" for i in range(n_classes)])
    if len(class_names) != n_classes:
        raise ValueError("class_names must follow y=0..n_classes-1")
    seeds = [primary_seed] + [int(s) for s in config.get("sensitivity_seeds", [11, 29]) if int(s) != primary_seed]
    expected_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    runs, files, run_hashes, exclusions = {}, {}, set(), []
    for path in sorted((root / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        key = (str(run["model"]), int(run["seed"]), int(run["perm"]))
        if key[0] not in model_names or key[1] not in seeds or key[2] < 0 or key[2] > expected:
            exclusions.append(str(path.relative_to(root)))
            continue
        if key[2] > 0 and key[1] != primary_seed:
            exclusions.append(str(path.relative_to(root)))
            continue
        if key in runs:
            raise ValueError(f"Duplicate completed task: {key}")
        if run.get("status", "complete") not in ("complete", "completed"):
            exclusions.append(str(path.relative_to(root)))
            continue
        if int(run.get("n_features", -1)) != 5520:
            raise ValueError(f"Unexpected model inputs in {path}")
        run_hashes.add(run["config_sha256"])
        runs[key] = run
        files[str(path.relative_to(root))] = sha256(path)
    if run_hashes and run_hashes != {expected_hash}:
        raise ValueError("Completed tasks do not match the current configuration hash")
    observed, importance_tables, performance, null_rows, fold_rows = {}, {}, [], [], []
    null_assignments = {}
    all_groups, all_importance_folds, all_subject_importance, all_stability = [], [], [], []
    confusion_rows = []
    for model in model_names:
        for seed in seeds:
            if model == "lda_full" and seed != primary_seed:
                continue  # Deterministic LDA has no neural training-seed sensitivity runs.
            run = runs.get((model, seed, 0))
            row = {"model": model, "seed": seed, "role": "primary" if seed == primary_seed else "sensitivity",
                   "status": "observed_pending", "balanced_accuracy_percent": None, "ci_low_percent": None,
                   "ci_high_percent": None, "n_subjects": None, "n_trials": None,
                   "null_completed": 0, "null_expected": expected if seed == primary_seed else 0,
                   "null_status": "pending" if seed == primary_seed else "not_requested",
                   "null_mean_percent": None, "null_95th_percentile_percent": None, "gain_pp": None,
                   "relative_gain_percent": None, "p_raw": None, "p_holm": None}
            score = None
            oof = None
            if run is not None:
                oof, order, path = load_oof(root, run, n_classes, n_folds,
                                            n_subjects, n_videos)
                files[str(path.relative_to(root))] = sha256(path)
                observed[(model, seed)] = oof
                summary = participant_bootstrap(oof["y"], oof["pred"], oof["subject"], n_classes, n_bootstrap, primary_seed)
                score = summary["balanced_accuracy"]
                row.update(status="complete", balanced_accuracy_percent=100 * score,
                           ci_low_percent=100 * summary["ci_low"], ci_high_percent=100 * summary["ci_high"],
                           n_subjects=summary["n_subjects"], n_trials=summary["n_trials"])
                for fold in np.unique(oof["fold"]):
                    take = oof["fold"] == fold
                    fold_rows.append({"model": model, "seed": seed, "fold": int(fold),
                                      "balanced_accuracy_percent": 100 * balanced_accuracy(oof["y"][take], oof["pred"][take], n_classes),
                                      "n_subjects": len(np.unique(oof["subject"][take])), "n_trials": int(take.sum())})
                cm = confusion_counts(oof["y"], oof["pred"], n_classes)
                for a, b in itertools.product(range(n_classes), repeat=2):
                    confusion_rows.append({"model": model, "seed": seed, "true_class": a, "true_label": class_names[a],
                                           "predicted_class": b, "predicted_label": class_names[b],
                                           "n_trials": int(cm[a, b]), "row_percent": 100 * cm[a, b] / cm[a].sum()})
                if run.get("importance_path"):
                    ipath = _within_root(root, run["importance_path"])
                    with np.load(ipath, allow_pickle=False) as data:
                        declared_groups = config.get("groups")
                        if declared_groups and (data["group_names"].tolist() != [g["name"] for g in declared_groups]
                                                or data["group_kinds"].tolist() != [g["kind"] for g in declared_groups]):
                            raise ValueError(f"Importance groups disagree with the fixed configuration: {ipath}")
                        n_repeats = data["predictions"].shape[1]
                        if n_repeats != int(config.get("importance_repeats", n_repeats)) or len(data["repeat_seeds"]) != n_repeats:
                            raise ValueError(f"Incomplete feature-permutation repetitions: {ipath}")
                        baseline = data["baseline_predictions"][order]
                        if not np.array_equal(baseline, oof["pred"]):
                            raise ValueError(f"Importance baseline disagrees with OOF predictions: {ipath}")
                        groups, folds, participants, stability = grouped_importance(
                            oof, data["predictions"][:, :, order], data["group_names"].tolist(),
                            data["group_kinds"].tolist(), n_classes, n_bootstrap, primary_seed)
                    files[str(ipath.relative_to(root))] = sha256(ipath)
                    for table in (groups, folds, participants, stability):
                        table.insert(0, "seed", seed)
                        table.insert(0, "model", model)
                    importance_tables[(model, seed)] = groups
                    all_groups.append(groups); all_importance_folds.append(folds)
                    all_subject_importance.append(participants); all_stability.append(stability)
            if seed == primary_seed:
                scores = {}
                for permutation in range(1, expected + 1):
                    null_run = runs.get((model, primary_seed, permutation))
                    if null_run is None:
                        continue
                    null, _, path = load_oof(root, null_run, n_classes, n_folds,
                                             n_subjects, n_videos)
                    if permutation in null_assignments:
                        for key in ("idx", "y"):
                            if not np.array_equal(null[key], null_assignments[permutation][key]):
                                raise ValueError("Models used different video-label assignments for the same permutation")
                    else:
                        null_assignments[permutation] = {key: null[key] for key in ("idx", "y")}
                    if oof is not None:
                        for key in ("idx", "subject", "video", "fold"):
                            if not np.array_equal(oof[key], null[key]):
                                raise ValueError(f"Permutation run changes the evaluation rows/folds: {path}")
                        if not np.array_equal(np.bincount(oof["y"], minlength=n_classes), np.bincount(null["y"], minlength=n_classes)):
                            raise ValueError(f"Permutation does not preserve class counts: {path}")
                    scores[permutation] = balanced_accuracy(null["y"], null["pred"], n_classes)
                    files[str(path.relative_to(root))] = sha256(path)
                    null_rows.append({"model": model, "seed": primary_seed, "permutation": permutation,
                                      "balanced_accuracy_percent": 100 * scores[permutation]})
                row.update(permutation_summary(score, scores, expected))
            performance.append(row)
    table = pd.DataFrame(performance)
    primary_mask = table.seed == primary_seed
    complete_primary = table[primary_mask]
    holm_complete = len(complete_primary) == len(model_names) and complete_primary.p_raw.notna().all()
    if holm_complete:
        table.loc[primary_mask, "p_holm"] = holm_adjust(complete_primary.p_raw.to_numpy(float))
    differences = []
    for seed in seeds:
        for a, b in itertools.combinations(model_names, 2):
            if (a, seed) not in observed or (b, seed) not in observed:
                continue
            differences.append({"model_a": a, "model_b": b, "seed": seed,
                                **paired_model_difference(observed[(a, seed)], observed[(b, seed)], n_classes, n_bootstrap, primary_seed)})
    destination = root / "report"
    destination.mkdir(parents=True, exist_ok=True)
    csvs = {"performance.csv": table, "fold_performance.csv": pd.DataFrame(fold_rows),
            "confusion_matrices.csv": pd.DataFrame(confusion_rows),
            "paired_model_differences.csv": pd.DataFrame(differences),
            "permutation_run_scores.csv": pd.DataFrame(null_rows)}
    for name, tables in [("feature_importance.csv", all_groups), ("feature_importance_folds.csv", all_importance_folds),
                         ("feature_importance_participants.csv", all_subject_importance),
                         ("feature_importance_rank_stability.csv", all_stability)]:
        csvs[name] = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    for name, frame in csvs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame.to_csv(destination / name, index=False)
    figures = make_figures(destination, table, observed, importance_tables, class_names, primary_seed, model_names)
    result = {"protocol": config.get("protocol", "participant-held-out classification of familiar video conditions"),
              "generalization_target": "unseen participants viewing the familiar stimulus set; not unseen videos",
              "primary_seed": primary_seed, "sensitivity_seeds": seeds[1:], "models": model_names,
              "training_epochs": config.get("epochs"), "importance_repeats": config.get("importance_repeats"),
              "n_bootstrap": n_bootstrap, "bootstrap_unit": "participant with all videos retained",
              "ci_scope": "conditional on fitted OOF models and sampled videos; importance averages saved feature permutations before participant resampling",
              "importance_scope": "model reliance under grouped feature permutation; not causal, not additive across overlapping groups; confidence intervals are pointwise descriptive intervals without multiplicity adjustment",
              "null_scheme": "one shared video-label assignment across participants per permutation, with full model refitting",
              "null_inference_policy": f"p=(1 + number of null scores >= observed)/{expected + 1}; requires all permutation IDs 1..{expected} for that model at the primary seed",
              "holm_scope": "all prespecified primary models; calculated only after every model's complete null is available",
              "holm_status": "complete" if holm_complete else "pending",
              "performance": table.to_dict("records"), "paired_model_differences": differences,
              "config_file_sha256": sha256(config_path), "config_run_sha256": expected_hash,
              "input_files_sha256": files, "excluded_run_files": exclusions,
              "report_script_sha256": sha256(Path(__file__)), "figures": figures}
    (destination / "report.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
    lines = ["Participant-held-out EEG prediction", "", "Generalization: unseen participants viewing familiar videos.",
             "Primary results use seed " + str(primary_seed) + "; seeds " + ", ".join(map(str, seeds[1:])) + " are separate sensitivity analyses.",
             "All three models receive the same 5,520 scalar inputs. No model ensemble is formed.", "",
             "Balanced accuracy is reported in percent; differences and feature importance are percentage points.",
             "95% intervals resample participants and retain all their trials. They condition on fitted OOF models",
             "and this stimulus set; they are not intervals for generalization to new videos.", "",
             "Permutation inference remains pending until every planned permutation completes for a model.",
             "No p-value or claim of above-null prediction is generated from a partial permutation sample.",
             "Holm adjustment across the prespecified models waits for all model nulls to complete.", "",
             "Feature importance is the OOF balanced-accuracy drop after permuting a feature group across",
             "videos within held-out participants, using a common video shuffle across participants. Repeats",
             "are averaged within participant before bootstrapping. Positive values indicate model reliance;",
             "negative values are retained. Correlated and overlapping groups are not additive, and importance",
             "does not establish causation or a unique physiological contribution. Intervals are pointwise",
             "descriptive intervals without multiplicity adjustment. Fold ranks are compared",
             "separately within the atomic and family levels. Rank correlations are descriptive across five folds.", "",
             "The overview's confusion matrix and feature panel use the prespecified hybrid GNN, not a model",
             "chosen after inspecting accuracy. All models' confusion matrices and importance tables are retained.", "",
             "This directory is an analysis report. No manuscript has been edited by this script."]
    (destination / "README.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report_directory": str(destination), "observed_runs": len(observed),
                      "holm_status": result["holm_status"], "figures": figures}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--expected-permutations", type=int, default=99)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    build_report(args.out_dir, args.expected_permutations, args.bootstrap)


if __name__ == "__main__":
    main()
