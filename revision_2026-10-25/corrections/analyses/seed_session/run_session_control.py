"""Post-audit SEED-VII source-held-out session controls.

Reproduce the published band-specific synchronization + power LDA exactly, then
compare training-only session frequencies and a restricted source-label null.
The restricted null permutes labels among single-condition sources with the
same vector of clip counts over the four recording sessions. Mixed-condition
sources and unique-profile sources remain fixed. This conditions on session
composition without splitting related excerpts. It tests conditional label
exchangeability only within the stated strata, not a universal emotion null.

All output is local to this directory; original cached data/results are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EMOTIONS = ["anger", "disgust", "fear", "sadness", "neutral", "happiness", "surprise"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_probability(y, probabilities):
    """Expected balanced accuracy if predictions are sampled from each row."""
    return float(np.mean([probabilities[y == c, c].mean() for c in range(7)]))


def fit_predict(X, y, folds):
    """Original LDA fitting, including training-only pooled standardization."""
    pred = np.empty_like(y)
    with threadpool_limits(limits=1):
        for _, train, test in folds:
            scaler = StandardScaler().fit(X[train])
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model.fit(scaler.transform(X[train]), y[train])
            pred[test] = model.predict(scaler.transform(X[test]))
    return pred


def permutation_fit(index, clip_labels, X, videos, folds):
    y = clip_labels[videos]
    pred = fit_predict(X, y, folds)
    return index, float(balanced_accuracy_score(y, pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    cache = ROOT / "seed7_replication/cache"
    df = pd.read_csv(cache / "features_scalar.csv")
    arrays = np.load(cache / "features_arrays.npz")
    assert len(df) == 1600 and not df.duplicated(["subject", "video"]).any()
    assert df.groupby("subject").video.nunique().eq(80).all()
    assert df.groupby("video").subject.nunique().eq(20).all()
    assert all(df.groupby("video")[c].nunique().eq(1).all()
               for c in ["emotion", "source", "session"])
    clips = df.groupby("video").first().reindex(range(80))
    assert clips.source.nunique() == 54
    videos = df.video.to_numpy(dtype=int)
    sessions = df.session.to_numpy(dtype=int)
    subjects = df.subject.to_numpy(dtype=int)
    raw = np.c_[df[[f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS]].values,
                np.hstack([arrays[f"logpow_{b}"] for b in BANDS])]
    # Exact original label-free participant normalization, ddof=1. This is
    # transductive preprocessing, fixed across observed and permuted labels.
    frame = pd.DataFrame(raw)
    grouped = frame.groupby(subjects)
    X = ((frame - grouped.transform("mean")) /
         (grouped.transform("std") + 1e-12)).to_numpy()
    assert X.shape == (1600, 160) and np.isfinite(X).all()
    clip_y = np.array([EMOTIONS.index(e) for e in clips.emotion])
    y = clip_y[videos]
    groups = clips.source.to_numpy()[videos]
    folds = [(str(k), np.flatnonzero(groups != k), np.flatnonzero(groups == k))
             for k in np.unique(groups)]
    fold_index = np.empty(len(df), dtype=int)
    fold_rows = []
    for i, (source, train, test) in enumerate(folds):
        fold_index[test] = i
        assert set(df.iloc[train].source).isdisjoint(df.iloc[test].source)
        fold_rows.append({"fold": i, "held_source": source,
                          "held_videos": json.dumps(sorted(df.iloc[test].video.unique().tolist())),
                          "n_train": len(train), "n_test": len(test)})
    pd.DataFrame(fold_rows).to_csv(out / "source_folds.csv", index=False)
    clips[["clip", "session", "emotion", "source", "title"]].to_csv(out / "clips.csv")
    np.savez_compressed(out / "primary_feature_snapshot.npz", raw=raw, normalized=X,
                        subject=subjects, video=videos, session=sessions, labels=y)

    started = time.monotonic()
    pred = fit_predict(X, y, folds)
    observed = float(balanced_accuracy_score(y, pred))
    original = pd.read_csv(ROOT / "seed7_replication/tables/SEED7_Table3_decoding.csv")
    original_score = float(original.loc[(original.folds == "leave-one-source-out") &
        (original.features == "Band-specific sync + power"), "balanced_accuracy"].iloc[0])
    assert np.isclose(observed, original_score, atol=1e-14, rtol=0)

    frequency = np.zeros((len(df), 7))
    ties = np.zeros_like(frequency)
    majority = np.empty_like(y)
    count_rows = []
    for i, (source, train, test) in enumerate(folds):
        for session in range(1, 5):
            train_s = train[sessions[train] == session]
            test_s = test[sessions[test] == session]
            counts = np.bincount(y[train_s], minlength=7)
            assert counts.sum() > 0
            freq = counts / counts.sum()
            maximum = counts == counts.max()
            # Probabilistic tie handling avoids dependence on arbitrary ordering.
            tie_prob = maximum / maximum.sum()
            frequency[test_s] = freq
            ties[test_s] = tie_prob
            majority[test_s] = counts.argmax()
            row = {"fold": i, "held_source": source, "session": session,
                   "n_test": len(test_s), "n_training": len(train_s),
                   "n_tied_majorities": int(maximum.sum())}
            row.update({f"train_{e}": int(counts[c]) for c, e in enumerate(EMOTIONS)})
            count_rows.append(row)
    assert np.allclose(frequency.sum(1), 1) and np.allclose(ties.sum(1), 1)
    pd.DataFrame(count_rows).to_csv(out / "training_session_condition_counts.csv", index=False)
    records = df[["subject", "video", "clip", "session", "emotion", "source"]].copy()
    records["fold"] = fold_index
    records["label"] = y
    records["lda_prediction"] = pred
    records["session_majority_prediction_first_tie"] = majority
    records["lda_correct"] = pred == y
    records["session_frequency_expected_correct"] = frequency[np.arange(len(y)), y]
    records["session_majority_tie_expected_correct"] = ties[np.arange(len(y)), y]
    for c, emotion in enumerate(EMOTIONS):
        records[f"session_frequency_probability_{emotion}"] = frequency[:, c]
        records[f"session_majority_tie_probability_{emotion}"] = ties[:, c]
    records.to_csv(out / "aligned_out_of_fold_predictions.csv", index=False)
    subject_rows = []
    for subject in np.unique(subjects):
        m = subjects == subject
        ba = float(balanced_accuracy_score(y[m], pred[m]))
        f = score_probability(y[m], frequency[m])
        t = score_probability(y[m], ties[m])
        subject_rows.append({"subject": int(subject), "lda_balanced_accuracy": ba,
                             "session_frequency_expected_balanced_accuracy": f,
                             "session_majority_tie_expected_balanced_accuracy": t,
                             "lda_minus_frequency_expected": ba - f,
                             "lda_minus_majority_tie_expected": ba - t})
    subject_table = pd.DataFrame(subject_rows)
    subject_table.to_csv(out / "participant_paired_scores.csv", index=False)
    assert np.isclose(subject_table.lda_balanced_accuracy.mean(), observed)
    rng_boot = np.random.default_rng(args.seed + 1)
    bootstrap_means = subject_table.lda_minus_frequency_expected.to_numpy()[
        rng_boot.integers(0, 20, size=(10000, 20))].mean(1)

    # Exchange labels only among complete pure sources with identical session
    # count vectors; this leaves class counts in EVERY session exactly unchanged.
    source_nconditions = clips.groupby("source").emotion.nunique()
    pure = source_nconditions[source_nconditions == 1].index
    mixed = source_nconditions[source_nconditions > 1].index.tolist()
    profile = pd.crosstab(clips.source, clips.session).reindex(columns=range(1, 5), fill_value=0)
    strata = {}
    for source in pure:
        key = tuple(int(x) for x in profile.loc[source])
        strata.setdefault(key, []).append(source)
    stratum_rows = []
    source_rows = []
    allocations = 1
    for key, sources in sorted(strata.items()):
        label_list = [int(clip_y[np.flatnonzero(clips.source.values == s)[0]]) for s in sources]
        multiplicities = np.bincount(label_list, minlength=7)
        unique = math.factorial(len(sources))
        for n in multiplicities:
            unique //= math.factorial(int(n))
        allocations *= unique
        stratum_rows.append({"session_count_profile": str(key), "n_sources": len(sources),
                             "n_distinct_labels": int(np.count_nonzero(multiplicities)),
                             "n_unique_label_allocations": str(unique),
                             "sources": json.dumps(sources, ensure_ascii=False)})
        for source, label in zip(sources, label_list):
            source_rows.append({"source": source, "session_count_profile": str(key),
                                "emotion": EMOTIONS[label], "permutable": unique > 1,
                                "reason_if_fixed": "" if unique > 1 else "unique session profile"})
    for source in mixed:
        source_rows.append({"source": source, "session_count_profile": str(tuple(profile.loc[source])),
                            "emotion": "mixed", "permutable": False,
                            "reason_if_fixed": "source contains two conditions"})
    pd.DataFrame(stratum_rows).to_csv(out / "permutation_strata.csv", index=False)
    source_table = pd.DataFrame(source_rows)
    source_table.to_csv(out / "source_exchangeability.csv", index=False)
    original_counts = np.stack([np.bincount(clip_y[clips.session.values == s], minlength=7)
                               for s in range(1, 5)])
    clip_maps = np.empty((args.n_permutations, 80), dtype=np.int16)
    rng = np.random.default_rng(args.seed)
    for n in range(args.n_permutations):
        permuted = clip_y.copy()
        for sources in strata.values():
            labels = [clip_y[np.flatnonzero(clips.source.values == s)[0]] for s in sources]
            for source, label in zip(sources, rng.permutation(labels)):
                permuted[clips.source.values == source] = label
        perm_counts = np.stack([np.bincount(permuted[clips.session.values == s], minlength=7)
                                for s in range(1, 5)])
        assert np.array_equal(original_counts, perm_counts)
        for source in pure:
            assert np.unique(permuted[clips.source.values == source]).size == 1
        for source in mixed:
            m = clips.source.values == source
            assert np.array_equal(permuted[m], clip_y[m])
        clip_maps[n] = permuted
    np.save(out / "restricted_permutation_clip_labels.npy", clip_maps)
    baseline = {
        "analysis_status": "observed complete; restricted permutations in progress",
        "classification_model": "Band-specific synchronization + power; shrinkage LDA",
        "n_subjects": 20, "n_clips": 80, "n_rows": 1600, "n_source_folds": 54,
        "observed_balanced_accuracy": observed,
        "original_saved_balanced_accuracy": original_score,
        "exact_original_reproduced_at_tolerance": 1e-14,
        "session_majority_first_tie_balanced_accuracy": float(balanced_accuracy_score(y, majority)),
        "session_majority_tie_averaged_expected_balanced_accuracy": score_probability(y, ties),
        "session_frequency_expected_balanced_accuracy": score_probability(y, frequency),
        "paired_lda_minus_frequency_expected": float(subject_table.lda_minus_frequency_expected.mean()),
        "paired_lda_minus_frequency_expected_conditional_participant_bootstrap_95":
            np.quantile(bootstrap_means, [.025, .975]).tolist(),
        "bootstrap_scope": "Descriptive participant resampling of fixed out-of-fold predictions; no model refit and no source sampling.",
        "majority_caveat": "Holding out a source depletes its condition within each training session; the observed held-out label is never among the session-frequency maxima. Zero majority accuracy does not remove session confounding.",
        "probabilistic_reference_caveat": "Expected balanced accuracy of sampling labels from training session frequencies, not measured accuracy of a fitted EEG classifier.",
        "permutation_seed": args.seed, "requested_permutations": args.n_permutations,
        "null_exchangeability": "Permute single-condition source labels within identical four-session clip-count profiles; all excerpts stay together. Mixed-condition and unique-profile sources fixed.",
        "n_exchangeable_sources": int(source_table.permutable.sum()),
        "n_fixed_sources": int((~source_table.permutable).sum()),
        "n_exchangeable_strata": sum(int(r["n_unique_label_allocations"]) > 1 for r in stratum_rows),
        "mixed_condition_sources_fixed": mixed,
        "n_unique_possible_label_allocations": str(allocations),
        "n_unique_draws": int(np.unique(clip_maps, axis=0).shape[0]),
        "session_condition_counts_invariant_all_draws": True,
        "same_source_excerpt_labels_preserved_all_draws": True,
        "preprocessing": "Original within-participant all-clip label-free normalization (ddof=1), then training-only pooled StandardScaler in each fold. Preprocessing and source folds identical across all label maps.",
        "post_audit_sensitivity": True,
        "inference_scope": "One additional model-specific conditional permutation sensitivity; no multiplicity claim, independent replication claim, or exclusion of all session/stimulus confounding.",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "input_sha256": {str(cache / name): sha(cache / name) for name in
                         ["features_scalar.csv", "features_arrays.npz", "clips.csv"]},
        "script_sha256": sha(__file__),
    }
    (out / "SESSION_CONTROL_SUMMARY.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(json.dumps({k: baseline[k] for k in ["observed_balanced_accuracy", "session_majority_first_tie_balanced_accuracy", "session_frequency_expected_balanced_accuracy", "n_exchangeable_sources", "n_unique_possible_label_allocations"]}), flush=True)
    null = np.full(args.n_permutations, np.nan)
    jobs = (delayed(permutation_fit)(i, labels, X, videos, folds) for i, labels in enumerate(clip_maps))
    results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(jobs)
    for done, (i, value) in enumerate(results, 1):
        null[i] = value
        if done % 20 == 0 or done == args.n_permutations:
            np.save(out / "restricted_null_balanced_accuracy_progress.npy", null)
            print(f"restricted permutations {done}/{args.n_permutations}; elapsed {time.monotonic() - started:.1f}s", flush=True)
    assert np.isfinite(null).all()
    np.save(out / "restricted_null_balanced_accuracy.npy", null)
    pd.DataFrame({"permutation": np.arange(args.n_permutations), "balanced_accuracy": null}).to_csv(
        out / "restricted_null_balanced_accuracy.csv", index=False)
    baseline.update({"analysis_status": "complete", "completed_permutations": len(null),
                     "restricted_null_mean": float(null.mean()),
                     "restricted_null_95th_percentile": float(np.percentile(null, 95)),
                     "restricted_null_exceedances": int((null >= observed).sum()),
                     "restricted_null_p_plus_one": float((1 + (null >= observed).sum()) / (1 + len(null))),
                     "gain_over_restricted_null_mean": float(observed - null.mean()),
                     "elapsed_seconds": time.monotonic() - started})
    (out / "SESSION_CONTROL_SUMMARY.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print(json.dumps({k: baseline[k] for k in ["analysis_status", "observed_balanced_accuracy", "restricted_null_mean", "restricted_null_95th_percentile", "restricted_null_p_plus_one", "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
