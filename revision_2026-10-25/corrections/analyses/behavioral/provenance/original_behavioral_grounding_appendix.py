#!/usr/bin/env python3
"""Generate a non-fabricated FACED behavioral-grounding supplement.

Inputs
------
1. Official NEMAR v1.1.3 participant events tables collected verbatim in a
   JSON mapping (sub-000 ... sub-122 -> TSV text).
2. The existing 3,444-file EEG trial export used by the manuscript.

The script validates the one-to-one subject x video merge, reproduces the
published synchronization definitions, computes participant-specific
behavioral analyses, and writes figures/tables/LaTeX text. It deliberately
does not analyse QG because the manuscript does not document the maximum lag
searched for QG construction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats
from sklearn.linear_model import LinearRegression


SEED = 20260901
FS = 250
CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2",
]
EMOTIONS = [
    "anger", "disgust", "fear", "sadness", "neutral",
    "amusement", "inspiration", "joy", "tenderness",
]
DISCRETE = [
    "Joy", "Tenderness", "Inspiration", "Amusement",
    "Anger", "Disgust", "Fear", "Sadness",
]
RATINGS = DISCRETE + ["Arousal", "Valence", "Familiarity", "Liking"]
METRICS = ["PLV", "PLI", "wPLI", "AEC", "MaxXC", "Lag"]
VIDEO_EMOTION = {}
for _emo, _a, _b in [
    ("anger", 1, 3), ("disgust", 4, 6), ("fear", 7, 9),
    ("sadness", 10, 12), ("neutral", 13, 16),
    ("amusement", 17, 19), ("inspiration", 20, 22),
    ("joy", 23, 25), ("tenderness", 26, 28),
]:
    for _v in range(_a, _b + 1):
        VIDEO_EMOTION[_v] = _emo

IU = np.triu_indices(len(CHANNELS), 1)
SOS = signal.butter(4, [1, 45], btype="bandpass", fs=FS, output="sos")


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    if not len(x):
        return out
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.clip(ranked, 0, 1)
    out[ok] = q
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_official_events(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {f"{i:03d}" for i in range(123)}:
        raise RuntimeError("Official events JSON must contain sub-000 through sub-122")
    rows = []
    qc = []
    for sid_text, tsv in sorted(raw.items()):
        d = pd.read_csv(io.StringIO(tsv), sep="\t", encoding="utf-8-sig")
        video = pd.to_numeric(d["video_index"], errors="coerce")
        d = d.loc[video.notna()].copy()
        d["video_id"] = video.loc[video.notna()].astype(int).to_numpy()
        d["subject_id"] = int(sid_text)
        for c in RATINGS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        expected = set(range(1, 29))
        valid = (
            len(d) == 28
            and set(d["video_id"]) == expected
            and not d["video_id"].duplicated().any()
            and not d[RATINGS].isna().any().any()
            and d[RATINGS].min().min() >= 0
            and d[RATINGS].max().max() <= 7
        )
        qc.append({
            "subject_id": int(sid_text), "n_video_rows": len(d),
            "n_unique_videos": d["video_id"].nunique(),
            "n_missing_ratings": int(d[RATINGS].isna().sum().sum()),
            "rating_min": float(d[RATINGS].min().min()),
            "rating_max": float(d[RATINGS].max().max()),
            "valid": bool(valid),
        })
        if not valid:
            raise RuntimeError(f"Invalid official events table for sub-{sid_text}")
        d["emotion"] = d["video_id"].map(VIDEO_EMOTION)
        rows.append(d[["subject_id", "video_id", "emotion"] + RATINGS])
    events = pd.concat(rows, ignore_index=True).sort_values(
        ["subject_id", "video_id"]
    ).reset_index(drop=True)
    if len(events) != 3444:
        raise RuntimeError(f"Expected 3,444 behavioral rows; found {len(events)}")
    return events, pd.DataFrame(qc)


def spearman_vec(x: np.ndarray) -> np.ndarray:
    ranked = np.apply_along_axis(stats.rankdata, 1, x)
    a = np.corrcoef(ranked)
    a = np.nan_to_num(np.abs(a), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(a, 0)
    return a[IU].astype(np.float32)


def mtf_vec(x: np.ndarray, q: int = 30) -> np.ndarray:
    n = x.shape[1]
    counts = np.zeros((q, q), float)
    for row in x:
        ranks = stats.rankdata(row, method="average")
        states = np.floor((ranks - 1) * q / n).astype(int)
        states = np.clip(states, 0, q - 1)
        np.add.at(counts, (states[:-1], states[1:]), 1)
    np.fill_diagonal(counts, 0)
    sums = counts.sum(1, keepdims=True)
    p = np.divide(counts, sums, out=np.zeros_like(counts), where=sums > 0)
    a = (p + p.T) / 2
    np.fill_diagonal(a, 0)
    return a[IU].astype(np.float32)


def synchronization(x: np.ndarray) -> np.ndarray:
    u = signal.sosfiltfilt(SOS, x, axis=1)
    z = signal.hilbert(u, axis=1)
    amp = np.abs(z)
    unit = z / np.maximum(amp, 1e-15)
    plv = np.abs(unit @ unit.conj().T / x.shape[1])
    im = np.imag(z[:, None, :] * np.conj(z[None, :, :]))
    pli = np.abs(np.mean(np.sign(im), axis=2))
    den = np.mean(np.abs(im), axis=2)
    wpli = np.divide(
        np.abs(np.mean(im, axis=2)), den,
        out=np.zeros_like(den), where=den > 0,
    )
    aec = np.corrcoef(amp)
    sd = u.std(1, keepdims=True)
    sd[sd == 0] = 1
    uz = (u - u.mean(1, keepdims=True)) / sd
    n = x.shape[1]
    f = np.fft.fft(uz, 512, axis=1)
    circ = np.fft.ifft(f[:, None, :] * np.conj(f[None, :, :]), axis=2).real
    full = np.concatenate([circ[:, :, -(n - 1):], circ[:, :, :n]], axis=2) / n
    idx = np.argmax(np.abs(full), axis=2)
    maxxc = np.take_along_axis(np.abs(full), idx[:, :, None], axis=2)[:, :, 0]
    lag_axis = np.arange(-(n - 1), n)
    lag = np.abs(lag_axis[idx]) * 1000 / FS
    values = [plv, pli, wpli, aec, maxxc, lag]
    return np.array([np.nanmean(v[IU]) for v in values], dtype=np.float32)


def one_eeg_file(path: Path):
    match = re.search(r"trial_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(path)
    video = int(match.group(1))
    folder = path.parts[-3].lower()
    d = pd.read_csv(
        path,
        usecols=["subject_id", "trial_id", "emotion_name"] + CHANNELS,
    )
    subject = int(d["subject_id"].iloc[0])
    trial_id = int(d["trial_id"].iloc[0])
    emotion = str(d["emotion_name"].iloc[0]).lower()
    if folder != emotion or VIDEO_EMOTION[video] != emotion or trial_id + 1 != video:
        raise RuntimeError(f"EEG stimulus mapping mismatch: {path}")
    x = d[CHANNELS].to_numpy(float).T
    if x.shape != (30, 250):
        raise RuntimeError(f"Expected 30 x 250 EEG matrix, got {x.shape}: {path}")
    return subject, video, emotion, spearman_vec(x), mtf_vec(x), synchronization(x)


def compute_features(raw_root: Path, out: Path, n_jobs: int):
    files = sorted(raw_root.glob("**/trial_*.csv"))
    if len(files) != 3444:
        raise RuntimeError(f"Expected 3,444 EEG files; found {len(files)}")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(one_eeg_file)(p) for p in files
    )
    subject = np.array([r[0] for r in results], np.int16)
    video = np.array([r[1] for r in results], np.int8)
    emotion = np.array([r[2] for r in results])
    spearman = np.stack([r[3] for r in results])
    mtf = np.stack([r[4] for r in results])
    sync = np.stack([r[5] for r in results])
    order = np.lexsort((video, subject))
    subject, video, emotion, spearman, mtf, sync = [
        a[order] for a in (subject, video, emotion, spearman, mtf, sync)
    ]
    cross = pd.crosstab(subject, video)
    if (
        len(np.unique(subject)) != 123 or len(np.unique(video)) != 28
        or cross.shape != (123, 28) or not (cross.to_numpy() == 1).all()
    ):
        raise RuntimeError("EEG export is not a complete 123 x 28 design")
    np.savez_compressed(
        out / "trial_network_features.npz", subject=subject, video=video,
        emotion=emotion, spearman=spearman, mtf=mtf, sync=sync,
    )
    scalar = pd.DataFrame({
        "subject_id": subject.astype(int), "video_id": video.astype(int),
        "emotion": emotion,
    })
    for j, metric in enumerate(METRICS):
        scalar[metric] = sync[:, j]
    scalar.to_csv(out / "trial_scalar_synchronization.csv", index=False)
    return subject, video, emotion, spearman, mtf, sync, scalar


def load_or_compute_features(raw_root: Path, out: Path, n_jobs: int):
    feature_path = out / "trial_network_features.npz"
    scalar_path = out / "trial_scalar_synchronization.csv"
    if feature_path.exists() and scalar_path.exists():
        z = np.load(feature_path, allow_pickle=False)
        scalar = pd.read_csv(scalar_path)
        return (
            z["subject"], z["video"], z["emotion"], z["spearman"],
            z["mtf"], z["sync"], scalar,
        )
    return compute_features(raw_root, out, n_jobs)


def within_subject_z(d: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = d.copy()
    qc = []
    for c in columns:
        mean = out.groupby("subject_id")[c].transform("mean")
        sd = out.groupby("subject_id")[c].transform(lambda x: x.std(ddof=0))
        zero_subjects = sorted(out.loc[sd <= 0, "subject_id"].unique().astype(int))
        z = np.divide(
            (out[c] - mean).to_numpy(float), sd.to_numpy(float),
            out=np.zeros(len(out), float), where=sd.to_numpy(float) > 0,
        )
        out[c + "_z"] = z
        qc.append({
            "metric": c,
            "n_subjects_zero_within_subject_sd": len(zero_subjects),
            "subject_ids_zero_within_subject_sd": ",".join(map(str, zero_subjects)),
            "standardization_rule": "z=0 when within-subject SD=0",
        })
    out.attrs["standardization_qc"] = pd.DataFrame(qc)
    return out


def cluster_ols(y: np.ndarray, x: np.ndarray, cluster: np.ndarray):
    """OLS with CR1 cluster-robust covariance and t inference by cluster."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]))
    groups = np.unique(cluster)
    for g in groups:
        idx = cluster == g
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    n, p = x.shape
    correction = (len(groups) / (len(groups) - 1)) * ((n - 1) / (n - p))
    cov = correction * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * stats.t.sf(np.abs(t), df=len(groups) - 1)
    return beta, se, t, pval, resid, np.linalg.matrix_rank(x)


def two_way_cluster_ols(
    y: np.ndarray, x: np.ndarray, cluster_a: np.ndarray, cluster_b: np.ndarray,
):
    """OLS with Cameron--Gelbach--Miller two-way clustered covariance.

    Finite-sample corrections are applied separately to participant, video,
    and participant-video intersection meats. Inference uses the conservative
    t reference df = min(G_participant, G_video) - 1.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    cluster_a = np.asarray(cluster_a)
    cluster_b = np.asarray(cluster_b)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    n, p = x.shape

    def meat_for(labels):
        meat = np.zeros((p, p))
        unique = np.unique(labels)
        for g in unique:
            idx = labels == g
            score = x[idx].T @ resid[idx]
            meat += np.outer(score, score)
        correction = (len(unique) / (len(unique) - 1)) * ((n - 1) / (n - p))
        return correction * meat, len(unique)

    meat_a, ga = meat_for(cluster_a)
    meat_b, gb = meat_for(cluster_b)
    intersection = np.array([
        f"{a}::{b}" for a, b in zip(cluster_a, cluster_b)
    ], dtype=object)
    meat_ab, _ = meat_for(intersection)
    cov = bread @ (meat_a + meat_b - meat_ab) @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    df = min(ga, gb) - 1
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2 * stats.t.sf(np.abs(t), df=df)
    return beta, se, t, pval, resid, np.linalg.matrix_rank(x), df


def fixed_effect_design(d: pd.DataFrame, predictors: list[str]):
    subject = pd.get_dummies(d["subject_id"].astype(str), drop_first=True, dtype=float)
    video = pd.get_dummies(d["video_id"].astype(str), drop_first=True, dtype=float)
    pred = d[predictors].astype(float).reset_index(drop=True)
    x = np.column_stack([
        np.ones(len(d)), pred.to_numpy(), subject.to_numpy(), video.to_numpy(),
    ])
    names = ["Intercept"] + predictors + [
        f"subject_{c}" for c in subject.columns
    ] + [f"video_{c}" for c in video.columns]
    return x, names


def standardize_within_video(d: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = d.copy()
    for c in columns:
        centered = out[c] - out.groupby("video_id")[c].transform("mean")
        sd = centered.std(ddof=0)
        out[c + "_wv_z"] = centered / sd
    return out


def primary_models(d: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = standardize_within_video(d, ["Valence", "Arousal", "Liking", "Familiarity"])
    rows = []
    sens = []
    primary_predictors = ["Valence_wv_z", "Arousal_wv_z"]
    sensitivity_predictors = primary_predictors + ["Liking_wv_z", "Familiarity_wv_z"]
    for metric in METRICS:
        y = d[metric + "_z"].to_numpy()
        x, names = fixed_effect_design(d, primary_predictors)
        beta, se, tval, pval, _, rank, df = two_way_cluster_ols(
            y, x, d.subject_id.to_numpy(), d.video_id.to_numpy()
        )
        for pred in primary_predictors:
            i = names.index(pred)
            rows.append({
                "metric": metric, "predictor": pred.replace("_wv_z", ""),
                "beta": beta[i], "se_two_way_cluster": se[i],
                "ci_low": beta[i] - stats.t.ppf(.975, df) * se[i],
                "ci_high": beta[i] + stats.t.ppf(.975, df) * se[i],
                "t": tval[i], "p": pval[i], "n_observations": len(d),
                "n_subjects": d.subject_id.nunique(),
                "n_videos": d.video_id.nunique(), "design_rank": rank,
                "outcome": "within-participant z score",
                "predictor_scale": "SD of within-video-centered rating",
                "fixed_effects": "participant and video",
                "se": "participant-and-video two-way clustered CGM",
                "reference_df": df,
            })
        xs, names_s = fixed_effect_design(d, sensitivity_predictors)
        b, s, tv, pv, _, rank_s, df_s = two_way_cluster_ols(
            y, xs, d.subject_id.to_numpy(), d.video_id.to_numpy()
        )
        for pred in sensitivity_predictors:
            i = names_s.index(pred)
            sens.append({
                "metric": metric, "predictor": pred.replace("_wv_z", ""),
                "beta": b[i], "se_two_way_cluster": s[i],
                "ci_low": b[i] - stats.t.ppf(.975, df_s) * s[i],
                "ci_high": b[i] + stats.t.ppf(.975, df_s) * s[i],
                "t": tv[i], "p": pv[i], "n_observations": len(d),
                "design_rank": rank_s, "reference_df": df_s,
            })
    result = pd.DataFrame(rows)
    result["q_BH_12"] = bh_fdr(result["p"].to_numpy())
    sensitivity = pd.DataFrame(sens)
    sensitivity["q_BH_24"] = bh_fdr(sensitivity["p"].to_numpy())
    result.to_csv(out / "Table_S7_valence_arousal_fixed_effects.csv", index=False)
    sensitivity.to_csv(out / "Table_S7b_liking_familiarity_sensitivity.csv", index=False)
    return result, sensitivity


def target_rating_models(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for emotion in [e for e in EMOTIONS if e != "neutral"]:
        q = d.loc[d.emotion == emotion].copy().reset_index(drop=True)
        rating = emotion.capitalize()
        centered = q[rating] - q.groupby("video_id")[rating].transform("mean")
        q["target_rating_wv_z"] = centered / centered.std(ddof=0)
        x, names = fixed_effect_design(q, ["target_rating_wv_z"])
        idx = names.index("target_rating_wv_z")
        for metric in METRICS:
            beta, se, tval, pval, _, rank = cluster_ols(
                q[metric + "_z"].to_numpy(), x, q.subject_id.to_numpy()
            )
            rows.append({
                "emotion": emotion, "rating": rating, "metric": metric,
                "beta": beta[idx], "se_cluster_subject": se[idx],
                "ci_low": beta[idx] - stats.t.ppf(.975, 122) * se[idx],
                "ci_high": beta[idx] + stats.t.ppf(.975, 122) * se[idx],
                "t": tval[idx], "p": pval[idx], "n_observations": len(q),
                "n_subjects": q.subject_id.nunique(),
                "n_videos": q.video_id.nunique(), "design_rank": rank,
            })
    result = pd.DataFrame(rows)
    result["q_BH_48"] = bh_fdr(result.p.to_numpy())
    result.to_csv(out / "Table_S8_target_emotion_rating_associations.csv", index=False)
    return result


def behavioral_validation(d: pd.DataFrame, out: Path):
    nonneutral = d[d.emotion != "neutral"].copy()
    discrete_lower = {c.lower(): c for c in DISCRETE}
    target = np.array([
        row[discrete_lower[row.emotion]] for _, row in nonneutral.iterrows()
    ], float)
    other = []
    target_is_max = []
    for _, row in nonneutral.iterrows():
        col = discrete_lower[row.emotion]
        vals = row[DISCRETE].to_numpy(float)
        other.append(np.mean([row[c] for c in DISCRETE if c != col]))
        target_is_max.append(row[col] >= np.max(vals) - 1e-12)
    nonneutral["target_rating"] = target
    nonneutral["mean_nontarget_rating"] = np.asarray(other)
    nonneutral["target_is_max_ties_included"] = target_is_max
    by_subject = nonneutral.groupby("subject_id").agg(
        target_rating=("target_rating", "mean"),
        mean_nontarget_rating=("mean_nontarget_rating", "mean"),
        target_is_max_rate=("target_is_max_ties_included", "mean"),
    )
    difference = by_subject.target_rating - by_subject.mean_nontarget_rating
    t, p = stats.ttest_1samp(difference, 0)
    rng = np.random.default_rng(SEED)
    boot = np.array([
        rng.choice(difference.to_numpy(), len(difference), replace=True).mean()
        for _ in range(10000)
    ])
    summary = [{
        "scope": "all non-neutral videos",
        "n_subjects": len(by_subject), "n_observations": len(nonneutral),
        "mean_target_rating": by_subject.target_rating.mean(),
        "mean_nontarget_rating": by_subject.mean_nontarget_rating.mean(),
        "mean_subject_level_difference": difference.mean(),
        "bootstrap_ci_low": np.quantile(boot, .025),
        "bootstrap_ci_high": np.quantile(boot, .975),
        "paired_subject_t": t, "paired_subject_p": p,
        "cohen_dz": difference.mean() / difference.std(ddof=1),
        "target_is_max_rate": by_subject.target_is_max_rate.mean(),
        "tie_rule": "target counted when tied for maximum",
    }]
    category_rows = []
    for emotion, q in nonneutral.groupby("emotion", sort=False):
        col = discrete_lower[emotion]
        category_rows.append({
            "emotion": emotion, "n_observations": len(q),
            "mean_target_rating": q[col].mean(),
            "sd_target_rating": q[col].std(ddof=1),
            "target_is_max_rate": q.target_is_max_ties_included.mean(),
            "mean_valence": q.Valence.mean(), "sd_valence": q.Valence.std(ddof=1),
            "mean_arousal": q.Arousal.mean(), "sd_arousal": q.Arousal.std(ddof=1),
        })
    pd.DataFrame(summary).to_csv(out / "Table_S6_behavioral_validation_overall.csv", index=False)
    pd.DataFrame(category_rows).to_csv(out / "Table_S6b_behavioral_validation_by_category.csv", index=False)
    heat = d.groupby("emotion")[DISCRETE].mean().reindex(EMOTIONS)
    heat.to_csv(out / "Table_S6c_mean_discrete_ratings_by_category.csv")
    return pd.DataFrame(summary), pd.DataFrame(category_rows), heat, nonneutral


def design_for_prediction(d: pd.DataFrame, model: str, train: np.ndarray, test: np.ndarray):
    # Participant and emotion one-hot columns are defined once to keep folds aligned.
    subject = pd.get_dummies(d.subject_id.astype(str), drop_first=True, dtype=float)
    emotion = pd.get_dummies(
        pd.Categorical(d.emotion, categories=EMOTIONS), drop_first=True, dtype=float
    )
    parts_train = [np.ones((train.sum(), 1)), subject.loc[train].to_numpy()]
    parts_test = [np.ones((test.sum(), 1)), subject.loc[test].to_numpy()]
    if model in ("label", "combined"):
        parts_train.append(emotion.loc[train].to_numpy())
        parts_test.append(emotion.loc[test].to_numpy())
    if model in ("rating", "combined"):
        rtrain = d.loc[train, ["Valence", "Arousal"]].to_numpy(float)
        rtest = d.loc[test, ["Valence", "Arousal"]].to_numpy(float)
        mean = rtrain.mean(0)
        sd = rtrain.std(0)
        parts_train.append((rtrain - mean) / sd)
        parts_test.append((rtest - mean) / sd)
    return np.column_stack(parts_train), np.column_stack(parts_test)


def leave_one_video_out(d: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for held_video in range(1, 29):
        test = d.video_id.eq(held_video).to_numpy()
        train = ~test
        designs = {
            m: design_for_prediction(d, m, train, test)
            for m in ["subject_only", "label", "rating", "combined"]
        }
        for metric in METRICS:
            y = d[metric + "_z"].to_numpy(float)
            for model, (x_train, x_test) in designs.items():
                fit = LinearRegression(fit_intercept=False).fit(x_train, y[train])
                pred = fit.predict(x_test)
                rmse = float(np.sqrt(np.mean((y[test] - pred) ** 2)))
                mae = float(np.mean(np.abs(y[test] - pred)))
                rows.append({
                    "held_video": held_video, "held_emotion": VIDEO_EMOTION[held_video],
                    "metric": metric, "model": model, "rmse": rmse, "mae": mae,
                    "n_train": int(train.sum()), "n_test": int(test.sum()),
                })
    folds = pd.DataFrame(rows)
    comparisons = []
    for metric in METRICS:
        q = folds[folds.metric == metric]
        wide = q.pivot(index="held_video", columns="model", values="rmse")
        for a, b in [("rating", "label"), ("combined", "label")]:
            diff = wide[a] - wide[b]
            test = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            comparisons.append({
                "metric": metric, "comparison": f"{a} minus {b}",
                "mean_rmse_difference": diff.mean(),
                "median_rmse_difference": diff.median(),
                "wilcoxon_statistic": test.statistic,
                "p": test.pvalue, "n_held_videos": len(diff),
                "interpretation": "negative favors first model",
            })
    comp = pd.DataFrame(comparisons)
    comp["q_BH_12"] = bh_fdr(comp.p.to_numpy())
    folds.to_csv(out / "Table_S9_leave_one_video_out_model_performance.csv", index=False)
    comp.to_csv(out / "Table_S9b_leave_one_video_out_comparisons.csv", index=False)
    return folds, comp


def row_correlation_rdm(x: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(x, float)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    zero = int(np.sum(norm[:, 0] <= 1e-15))
    norm[norm <= 1e-15] = np.nan
    z = x / norm
    similarity = z @ z.T
    return 1 - similarity, zero


def partial_spearman(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(covariate)
    xr = stats.rankdata(x[ok])
    yr = stats.rankdata(y[ok])
    c = np.column_stack([np.ones(ok.sum()), stats.rankdata(covariate[ok])])
    bx = np.linalg.lstsq(c, xr, rcond=None)[0]
    by = np.linalg.lstsq(c, yr, rcond=None)[0]
    return float(np.corrcoef(xr - c @ bx, yr - c @ by)[0, 1])


def rsa_analysis(
    d: pd.DataFrame, subject: np.ndarray, video: np.ndarray,
    spearman: np.ndarray, mtf: np.ndarray, out: Path,
):
    tri = np.triu_indices(28, 1)
    categorical = np.array([
        0.0 if VIDEO_EMOTION[i + 1] == VIDEO_EMOTION[j + 1] else 1.0
        for i, j in zip(*tri)
    ])
    rows = []
    behavior_rdms, sp_rdms, mtf_rdms = [], [], []
    zero_profiles = 0
    zero_profile_rows = []
    for sid in range(123):
        behavior = d.loc[d.subject_id == sid].sort_values("video_id")[DISCRETE].to_numpy(float)
        b_rdm, zeros = row_correlation_rdm(behavior)
        zero_profiles += zeros
        zero_idx = np.where(np.linalg.norm(
            behavior - behavior.mean(axis=1, keepdims=True), axis=1
        ) <= 1e-15)[0]
        for idx_zero in zero_idx:
            zero_profile_rows.append({
                "subject_id": sid, "video_id": int(idx_zero + 1),
                "reason": "constant eight-emotion rating profile; correlation distance undefined",
            })
        idx = subject == sid
        order = np.argsort(video[idx])
        sp_rdm, _ = row_correlation_rdm(spearman[idx][order])
        mtf_rdm, _ = row_correlation_rdm(mtf[idx][order])
        behavior_rdms.append(b_rdm)
        sp_rdms.append(sp_rdm)
        mtf_rdms.append(mtf_rdm)
        b = b_rdm[tri]
        for representation, nr in [("Spearman", sp_rdm), ("MTF", mtf_rdm)]:
            n = nr[tri]
            ok = np.isfinite(b) & np.isfinite(n)
            rho = stats.spearmanr(b[ok], n[ok]).statistic
            partial = partial_spearman(b, n, categorical)
            rows.append({
                "subject_id": sid, "representation": representation,
                "spearman_rho": rho, "partial_rho_controlling_assigned_category": partial,
                "n_rdm_pairs": int(ok.sum()),
            })
    subj = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    summary = []
    for representation in ["Spearman", "MTF"]:
        q = subj[subj.representation == representation]
        for column, label in [
            ("spearman_rho", "raw"),
            ("partial_rho_controlling_assigned_category", "category-adjusted"),
        ]:
            values = q[column].to_numpy(float)
            boot = np.array([
                rng.choice(values, len(values), replace=True).mean()
                for _ in range(10000)
            ])
            observed = values.mean()
            null = np.array([
                np.mean(values * rng.choice([-1, 1], len(values)))
                for _ in range(10000)
            ])
            p = (1 + np.sum(np.abs(null) >= abs(observed))) / 10001
            summary.append({
                "representation": representation, "analysis": label,
                "mean_rho": observed, "median_rho": np.median(values),
                "sd_rho": values.std(ddof=1),
                "bootstrap_ci_low": np.quantile(boot, .025),
                "bootstrap_ci_high": np.quantile(boot, .975),
                "sign_flip_p": p, "n_subjects": len(values),
                "sign_flip_repetitions": 10000,
                "n_constant_behavioral_profiles_excluded": zero_profiles,
            })
    summary = pd.DataFrame(summary)
    summary["q_BH_4"] = bh_fdr(summary.sign_flip_p.to_numpy())
    subj.to_csv(out / "Table_S10_participant_behavioral_neural_RSA.csv", index=False)
    summary.to_csv(out / "Table_S10b_behavioral_neural_RSA_summary.csv", index=False)
    pd.DataFrame(zero_profile_rows).to_csv(
        out / "Table_S10c_RSA_undefined_behavioral_profiles.csv", index=False
    )
    means = {
        "behavior": np.nanmean(np.stack(behavior_rdms), axis=0),
        "spearman": np.nanmean(np.stack(sp_rdms), axis=0),
        "mtf": np.nanmean(np.stack(mtf_rdms), axis=0),
    }
    np.savez_compressed(out / "group_mean_rdms.npz", **means)
    return subj, summary, means


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, letter):
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)


def save_figure(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_s22(d, heat, category, out):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(heat.to_numpy(), vmin=0, vmax=7, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(DISCRETE)))
    ax.set_xticklabels(DISCRETE, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels([e.capitalize() for e in EMOTIONS])
    ax.set_title("Mean participant-reported discrete emotions")
    fig.colorbar(im, ax=ax, fraction=.035, pad=.02, label="Rating (0–7)")
    panel(ax, "a")
    colors = ["#C85A5A"] * 4 + ["#8C8C8C"] + ["#4C78A8"] * 4
    for ax, variable, letter in [(axes[0, 1], "Valence", "b"), (axes[1, 0], "Arousal", "c")]:
        groups = [d.loc[d.emotion == e, variable].to_numpy() for e in EMOTIONS]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=.65)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(.75); patch.set_linewidth(.6)
        for element in ["whiskers", "caps", "medians"]:
            for item in bp[element]: item.set_linewidth(.6)
        ax.set_xticks(range(1, 10))
        ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=45, ha="right")
        ax.set_ylim(-.15, 7.15); ax.set_ylabel(f"{variable} rating (0–7)")
        ax.set_title(f"Participant-reported {variable.lower()}")
        panel(ax, letter)
    ax = axes[1, 1]
    order = [e for e in EMOTIONS if e != "neutral"]
    values = category.set_index("emotion").loc[order, "target_is_max_rate"]
    ax.barh(range(len(order)), values, color=[colors[EMOTIONS.index(e)] for e in order])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([e.capitalize() for e in order])
    ax.set_xlim(0, 1); ax.set_xlabel("Target rating tied for highest (proportion)")
    ax.set_title("Participant–video label/rating concordance")
    ax.invert_yaxis(); panel(ax, "d")
    save_figure(fig, out, "Supplementary_Figure_S22_behavioral_validation")


def figure_s23(primary, sensitivity, folds, out):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), constrained_layout=True)
    for ax, table, title, letter, predictors in [
        (axes[0], primary, "Primary within-video model", "a", ["Valence", "Arousal"]),
        (axes[1], sensitivity, "Sensitivity model", "b", ["Valence", "Arousal", "Liking", "Familiarity"]),
    ]:
        y = 0
        ticks, labels = [], []
        palette = {"Valence":"#4C78A8", "Arousal":"#E45756", "Liking":"#72B7B2", "Familiarity":"#B279A2"}
        for metric in METRICS:
            q = table[table.metric == metric].set_index("predictor")
            for j, pred in enumerate(predictors):
                row = q.loc[pred]
                yy = y + (j - (len(predictors)-1)/2) * .16
                ax.errorbar(row.beta, yy, xerr=[[row.beta-row.ci_low],[row.ci_high-row.beta]],
                            fmt="o", color=palette[pred], ms=3, lw=.8, capsize=1.5)
            ticks.append(y); labels.append(metric); y += 1
        ax.axvline(0, color="black", lw=.6, ls="--")
        ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel("Standardized association (β ± 95% CI)")
        ax.set_title(title); panel(ax, letter)
        handles = [plt.Line2D([0],[0],marker='o',linestyle='',color=palette[p],label=p,markersize=4) for p in predictors]
        ax.legend(handles=handles, frameon=False, fontsize=6, loc="best")
    ax = axes[2]
    means = folds.groupby(["metric", "model"]).rmse.mean().unstack()
    models = ["subject_only", "label", "rating", "combined"]
    x = np.arange(len(METRICS)); width = .19
    model_colors = ["#BDBDBD", "#F2A541", "#4C78A8", "#59A14F"]
    for j, model in enumerate(models):
        ax.bar(x + (j-1.5)*width, means[model], width, label=model.replace("_", " "), color=model_colors[j])
    ax.set_xticks(x); ax.set_xticklabels(METRICS, rotation=45, ha="right")
    ax.set_ylabel("Mean held-video RMSE")
    ax.set_title("Leave-one-video-out prediction")
    ax.legend(frameon=False, fontsize=5.5, loc="lower center", ncol=2)
    panel(ax, "c")
    save_figure(fig, out, "Supplementary_Figure_S23_behavioral_EEG_models")


def figure_s24(subj, means, out):
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), constrained_layout=True)
    for ax, key, title, letter in [
        (axes[0], "behavior", "Behavioral RDM", "a"),
        (axes[1], "spearman", "Spearman-network RDM", "b"),
        (axes[2], "mtf", "MTF RDM", "c"),
    ]:
        im = ax.imshow(means[key], cmap="magma", aspect="equal")
        ax.set_title(title); ax.set_xlabel("Video index"); ax.set_ylabel("Video index")
        ax.set_xticks([0, 6, 13, 20, 27]); ax.set_xticklabels([1, 7, 14, 21, 28])
        ax.set_yticks([0, 6, 13, 20, 27]); ax.set_yticklabels([1, 7, 14, 21, 28])
        fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
        panel(ax, letter)
    ax = axes[3]
    labels, values, colors = [], [], []
    for representation, color in [("Spearman", "#4C78A8"), ("MTF", "#E45756")]:
        q = subj[subj.representation == representation]
        labels += [f"{representation}\nraw", f"{representation}\nadjusted"]
        values += [q.spearman_rho.to_numpy(), q.partial_rho_controlling_assigned_category.to_numpy()]
        colors += [color, color]
    vp = ax.violinplot(values, showmedians=True, showextrema=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.55)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(.8)
    ax.axhline(0, color="black", lw=.6, ls="--")
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Participant-level RSA Spearman ρ")
    ax.set_title("Behavioral–neural geometry")
    panel(ax, "d")
    save_figure(fig, out, "Supplementary_Figure_S24_behavioral_neural_RSA")


def format_p(p):
    if p < .001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def write_latex(
    out: Path, overall: pd.DataFrame, primary: pd.DataFrame,
    target: pd.DataFrame, folds: pd.DataFrame, loov_comparisons: pd.DataFrame,
    rsa_summary: pd.DataFrame,
):
    o = overall.iloc[0]
    sig = primary[primary.q_BH_12 < .05].sort_values("q_BH_12")
    sig_text = (
        "; ".join(
            f"{r.metric}--{r.predictor.lower()} ($\\beta={r.beta:.3f}$, "
            f"95\\% CI [{r.ci_low:.3f}, {r.ci_high:.3f}], "
            f"$q={format_p(r.q_BH_12)}$)" for r in sig.itertuples()
        ) if len(sig) else "No valence or arousal coefficient survived the 12-test BH--FDR correction"
    )
    strongest = primary.sort_values("p").iloc[0]
    target_sig = target[target.q_BH_48 < .05]
    target_text = (
        f"{len(target_sig)} of 48 coefficients survived BH--FDR correction"
        if len(target_sig)
        else "None of the 48 emotion-specific target-rating coefficients survived BH--FDR correction"
    )
    loov_sig = loov_comparisons[loov_comparisons.q_BH_12 < .05]
    loov_text = (
        f"{len(loov_sig)} of 12 paired model comparisons survived BH--FDR correction"
        if len(loov_sig)
        else "None of the 12 paired held-video model comparisons survived BH--FDR correction"
    )
    rmse = folds.groupby(["metric", "model"]).rmse.mean().unstack()
    rmse_lines = "; ".join(
        f"{m}: label {rmse.loc[m,'label']:.3f}, rating {rmse.loc[m,'rating']:.3f}, combined {rmse.loc[m,'combined']:.3f}"
        for m in METRICS
    )
    rsa_lines = "; ".join(
        f"{r.representation} {r.analysis}: mean $\\rho={r.mean_rho:.3f}$, "
        f"95\\% CI [{r.bootstrap_ci_low:.3f}, {r.bootstrap_ci_high:.3f}], "
        f"$q={format_p(r.q_BH_4)}$" for r in rsa_summary.itertuples()
    )
    text = rf"""\section{{Behavioral grounding of affective-condition EEG network organization}}
\label{{app:behavioral-grounding}}

\subsection{{Behavioral data provenance and merge audit}}

To distinguish experimenter-assigned stimulus categories from participants' reported affective experience, we used the participant-level event metadata from the official FACED NEMAR release v1.1.3. FACED provides 12 ratings after each video: joy, tenderness, inspiration, amusement, anger, disgust, fear, sadness, arousal, valence, familiarity, and liking, each on a 0--7 scale. These variables and their interpretation follow the dataset article, \emph{{A Large Finer-grained Affective Computing EEG Dataset}} \citep{{Chen2023FACED}}, and the official NEMAR metadata release \citep{{NEMARFACED2026}}.

The behavioral tables were filtered to rows with numeric video indices and were required to contain exactly one complete row for every video 1--28 for every participant. All 123 participants passed this audit, yielding 3,444 participant--video observations with no missing rating values and observed ratings spanning the documented 0--7 range. The uploaded EEG export also contained exactly 3,444 files. Its zero-based subject identifier, global trial identifier, folder-level emotion name, and filename-level video identifier were mutually consistent, allowing a validated one-to-one merge on participant and video identity.

The neural export used in the manuscript contains one 250-sample (1-s at 250 Hz) EEG segment per participant--video pair, whereas the behavioral rating was collected after the complete video. Consequently, the analyses below test whether network organization in the exported 1-s segment is associated with the participant's subsequent whole-video rating. They must not be described as analyses of the entire video-evoked EEG response.

\subsection{{Behavioral validation of the assigned categories}}

Across the eight non-neutral assigned categories, the participant-level mean target-emotion rating exceeded the mean of the seven non-target ratings by {o.mean_subject_level_difference:.3f} rating units (participant-bootstrap 95\% CI [{o.bootstrap_ci_low:.3f}, {o.bootstrap_ci_high:.3f}]; paired participant test $t_{{122}}={o.paired_subject_t:.2f}$, $p={format_p(o.paired_subject_p)}$; $d_z={o.cohen_dz:.2f}$). The assigned target rating was tied for the largest of the eight discrete-emotion ratings in {100*o.target_is_max_rate:.1f}\% of non-neutral participant--video observations. Supplementary Fig.~\ref{{fig:supp-behavioral-validation}} shows the full assigned-category by reported-emotion matrix and the participant-level valence and arousal distributions. These descriptive results validate the intended manipulation while also retaining the individual variability required for within-video analyses.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{Supplementary_Figure_S22_behavioral_validation.pdf}}
\caption{{\textbf{{Behavioral validation of the FACED affective stimulus conditions.}} \textbf{{a,}} Mean participant-reported ratings for eight discrete emotions within each assigned stimulus category. \textbf{{b,c,}} Participant-level valence and arousal distributions by assigned category. Boxes show medians and interquartile ranges; whiskers use the standard 1.5$\times$IQR rule and outliers are omitted from the display only. \textbf{{d,}} Proportion of participant--video observations in which the assigned target-emotion rating was tied for the maximum of the eight discrete ratings. Neutral videos have no target discrete-emotion rating and are therefore excluded from panel d.}}
\label{{fig:supp-behavioral-validation}}
\end{{figure}}

\subsection{{Within-video valence and arousal associations}}

For each synchronization outcome, values were standardized within participant across the 28 videos. Valence and arousal were centered within video and then scaled by their across-observation standard deviation. We fitted ordinary least-squares models containing participant and video fixed effects. Standard errors were clustered jointly by participant and video using the two-way Cameron--Gelbach--Miller covariance estimator, with the smaller cluster dimension determining the reference degrees of freedom ($df=27$). This design estimates whether participants who rated the same video as relatively more positive/negative or more/less arousing also differed in the synchronization measure, after removing stable participant differences and video-level mean differences. BH--FDR correction was applied jointly to the 12 valence/arousal coefficients (six outcomes $\times$ two predictors).

{sig_text}. The smallest uncorrected primary-model result was {strongest.metric}--{strongest.predictor.lower()} ($\beta={strongest.beta:.3f}$, 95\% CI [{strongest.ci_low:.3f}, {strongest.ci_high:.3f}], $p={format_p(strongest.p)}$, $q={format_p(strongest.q_BH_12)}$).

A sensitivity model additionally included within-video-centered liking and familiarity. Complete coefficients are reported in Supplementary Table~S7b; arousal was not relabeled as salience because FACED does not provide a direct salience measure.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{Supplementary_Figure_S23_behavioral_EEG_models.pdf}}
\caption{{\textbf{{Behaviorally grounded models of EEG synchronization.}} \textbf{{a,}} Participant- and video-fixed-effect estimates for within-video-centered valence and arousal. Outcomes are within-participant standardized synchronization measures; predictors are scaled within-video rating deviations. Intervals are participant- and video-clustered 95\% confidence intervals. \textbf{{b,}} Sensitivity model additionally controlling for liking and familiarity. \textbf{{c,}} Mean root-mean-square error across 28 leave-one-video-out folds for participant-only, assigned-label, rating-only, and combined models. Lower values indicate better held-video prediction.}}
\label{{fig:supp-behavioral-models}}
\end{{figure}}

\subsection{{Assigned labels versus reported experience}}

To compare assigned labels with reported experience without reusing the same video in training and testing, we performed 28 leave-one-video-out folds. Each model included participant indicators. The label model added assigned emotion; the rating model added valence and arousal; and the combined model included both. Rating variables were standardized using the training fold only. Mean held-video RMSE values were {rmse_lines}. {loov_text}. Fold-level results and paired comparisons are reported in Supplementary Tables~S9 and S9b. These model comparisons are predictive sensitivity analyses; they do not turn arousal into a measure of salience and do not remove the limitation that the neural input is a selected 1-s segment.

As an exploratory discrete-emotion analysis, each non-neutral category was analysed separately using its corresponding participant-reported target-emotion score, centered within video. Because each category contained only three videos, these models used participant-clustered standard errors and are not presented as video-clustered confirmatory tests. {target_text}; complete coefficients are reported in Supplementary Table~S8.

\subsection{{Behavioral--neural representational similarity}}

For each participant, a 28$\times$28 behavioral representational dissimilarity matrix (RDM) was constructed as one minus the Pearson correlation between the eight-dimensional discrete-emotion rating profiles of each video pair. Analogous RDMs were constructed from the 435 Spearman channel-network edges and the 435 node-matched $q=30$ MTF edges. Behavioral and neural RDM vectors were compared using Spearman correlation within participant; inference was then performed across the 123 participant-level coefficients using 10,000 sign-flip permutations and 10,000 participant bootstraps. A secondary partial RSA residualized both ranked RDM vectors with respect to an assigned-category same/different RDM. QG was not recomputed because the maximum lag searched for QG construction is not documented in the current manuscript.

Five participant--video observations had constant eight-emotion rating profiles, for which correlation distance is undefined. Video pairs involving those profiles were excluded from the corresponding participant-level RSA; no values were imputed and the retained RDM-pair count is reported for every participant in Supplementary Table~S10.

{rsa_lines}.

\begin{{figure}}[!htbp]
\centering
\includegraphics[width=\textwidth]{{Supplementary_Figure_S24_behavioral_neural_RSA.pdf}}
\caption{{\textbf{{Behavioral--neural representational similarity.}} \textbf{{a--c,}} Participant-averaged behavioral, Spearman-network, and MTF representational dissimilarity matrices across the 28 videos. \textbf{{d,}} Participant-level behavioral--neural RSA coefficients before and after residualizing assigned-category same/different structure. Violin plots display the complete distribution across 123 participants; horizontal bars mark medians.}}
\label{{fig:supp-behavioral-rsa}}
\end{{figure}}

\subsection{{Interpretational boundary}}

These analyses directly test participant-reported valence, arousal, and discrete-emotion profiles and therefore reduce reliance on experimenter-assigned labels. They do not establish that arousal is equivalent to salience, and they do not convert the selected 1-s EEG segment into a measure of the full video response. The strongest defensible interpretation is therefore that synchronization and network geometry in the analysed EEG export may covary with participants' reported affective experience to the extent supported by the coefficients and corrected tests above; null or weak results must be retained as such.
"""
    (out / "Appendix_J_behavioral_grounding.tex").write_text(text, encoding="utf-8")


def write_references(out: Path):
    text = r"""@article{Chen2023FACED,
  title = {{A Large Finer-grained Affective Computing EEG Dataset}},
  author = {Chen, Jingjing and Wang, Xiaobin and Huang, Chen and Hu, Xin and Shen, Xinke and Zhang, Dan},
  journal = {Scientific Data},
  volume = {10},
  pages = {740},
  year = {2023},
  doi = {10.1038/s41597-023-02650-w}
}

@misc{NEMARFACED2026,
  title = {{FACED - Finer-grained Affective Computing EEG Dataset}},
  author = {Liu, Yisi and Sourina, Olga and Nguyen, Minh Khoa},
  howpublished = {NEMAR, version v1.1.3},
  year = {2026},
  note = {doi: 10.82901/nemar.nm000112}
}
"""
    (out / "behavioral_grounding_references.bib").write_text(text, encoding="utf-8")


def write_readme(out: Path, manifest: dict):
    text = f"""# FACED behavioral-grounding supplement

This package was generated from the official FACED/NEMAR v1.1.3 participant
event tables and the exact 3,444-file EEG export used by the manuscript.

Dataset article: Jingjing Chen et al., *A Large Finer-grained Affective
Computing EEG Dataset*, Scientific Data 10, 740 (2023),
https://doi.org/10.1038/s41597-023-02650-w.

Behavioral metadata release: FACED - Finer-grained Affective Computing EEG
Dataset, NEMAR v1.1.3 (2026), https://doi.org/10.82901/nemar.nm000112.

## Hard validation results

- Participants: {manifest['n_subjects']}
- Videos per participant: {manifest['n_videos']}
- Behavioral rows: {manifest['n_behavioral_rows']}
- EEG trial files: {manifest['n_eeg_files']}
- EEG samples per participant-video file: {manifest['samples_per_eeg_file']}
- Missing behavioral ratings: {manifest['missing_behavioral_ratings']}
- Merge cardinality: one-to-one on subject_id + video_id

## Interpretational limit

Each behavioral score was reported after the complete video, but the uploaded
neural export contains one 1-second EEG segment per participant-video pair.
Results therefore relate that selected 1-second segment to the subsequent
whole-video rating; they do not summarize the full video-evoked EEG response.

FACED does not include a variable explicitly measuring salience. Arousal is
analysed as arousal and is never relabeled as salience.

QG is not recomputed because the current manuscript does not state the maximum
lag searched during QG construction. No undocumented value is inserted.

## Main outputs

- `Behavioral_grounding_appendix_standalone.pdf`: compiled, visually verified
  appendix preview.
- `Appendix_J_behavioral_grounding.tex`: manuscript-ready section populated
  only with computed values.
- `behavioral_grounding_references.bib`: verified dataset article and release
  citations used by the appendix.
- `Supplementary_Figure_S22_behavioral_validation.*`
- `Supplementary_Figure_S23_behavioral_EEG_models.*`
- `Supplementary_Figure_S24_behavioral_neural_RSA.*`
- Tables S6-S10 as CSV files.
- `analysis_manifest.json`: provenance, hashes, dimensions, and analysis rules.
- `behavioral_grounding_appendix.py` and
  `FACED_official_events_v1.1.3.json`: complete analysis code and the exact
  validated official event tables used as behavioral input.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-json", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--eeg-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    events, qc = read_official_events(args.events_json)
    qc.to_csv(args.out / "behavioral_source_validation_by_subject.csv", index=False)
    subject, video, emotion, sp, mtf, sync, scalar = load_or_compute_features(
        args.raw_root, args.out, args.n_jobs
    )
    merged = scalar.merge(
        events, on=["subject_id", "video_id", "emotion"],
        how="inner", validate="one_to_one",
    )
    if len(merged) != 3444:
        raise RuntimeError(f"Expected 3,444 merged rows; found {len(merged)}")
    merged = within_subject_z(merged, METRICS)
    standardization_qc = merged.attrs["standardization_qc"].copy()
    standardization_qc.to_csv(
        args.out / "synchronization_standardization_qc.csv", index=False
    )
    merged.to_csv(args.out / "behavioral_EEG_merged_validated.csv", index=False)

    overall, category, heat, nonneutral = behavioral_validation(merged, args.out)
    primary, sensitivity = primary_models(merged, args.out)
    target = target_rating_models(merged, args.out)
    folds, loov_comparisons = leave_one_video_out(merged, args.out)
    rsa_subj, rsa_summary, mean_rdms = rsa_analysis(
        merged, subject, video, sp, mtf, args.out
    )

    figure_s22(merged, heat, category, args.out)
    figure_s23(primary, sensitivity, folds, args.out)
    figure_s24(rsa_subj, mean_rdms, args.out)
    write_latex(
        args.out, overall, primary, target, folds, loov_comparisons, rsa_summary
    )
    write_references(args.out)

    manifest = {
        "analysis_name": "FACED behavioral grounding",
        "created_by_script": Path(__file__).name,
        "seed": SEED,
        "official_behavioral_source": "FACED NEMAR v1.1.3 events.tsv",
        "official_dataset_doi": "10.82901/nemar.nm000112",
        "dataset_article_title": "A Large Finer-grained Affective Computing EEG Dataset",
        "dataset_article_doi": "10.1038/s41597-023-02650-w",
        "n_subjects": 123, "n_videos": 28,
        "n_behavioral_rows": len(events), "n_eeg_files": len(scalar),
        "samples_per_eeg_file": 250, "sampling_rate_hz": 250,
        "segment_duration_seconds": 1,
        "missing_behavioral_ratings": int(events[RATINGS].isna().sum().sum()),
        "rating_range_observed": [float(events[RATINGS].min().min()), float(events[RATINGS].max().max())],
        "merge_keys": ["subject_id", "video_id", "emotion"],
        "merge_validation": "one_to_one; 3444 rows",
        "primary_model": "participant + video fixed effects; participant-and-video two-way clustered Cameron-Gelbach-Miller SE; reference df 27",
        "primary_multiple_testing_family": "6 outcomes x 2 predictors = 12 BH-FDR tests",
        "rsa_inference": "participant-level coefficients; 10000 sign flips; 10000 participant bootstraps",
        "rsa_constant_behavioral_profiles_excluded": int(
            rsa_summary.n_constant_behavioral_profiles_excluded.iloc[0]
        ),
        "qg": "not recomputed because maximum lag search is undocumented",
        "salience": "not operationalized; FACED provides arousal, not a direct salience measure",
        "critical_limit": "one 1-second EEG segment is linked to a whole-video post-stimulus rating",
        "zero_within_subject_sd_by_metric": {
            r.metric: int(r.n_subjects_zero_within_subject_sd)
            for r in standardization_qc.itertuples()
        },
        "events_json_sha256": sha256(args.events_json),
        "eeg_zip_sha256": sha256(args.eeg_zip),
    }
    (args.out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(args.out, manifest)
    print(json.dumps({
        "output": str(args.out),
        "n_merged": len(merged),
        "n_primary_fdr_significant": int((primary.q_BH_12 < .05).sum()),
        "n_target_fdr_significant": int((target.q_BH_48 < .05).sum()),
        "n_rsa_fdr_significant": int((rsa_summary.q_BH_4 < .05).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
