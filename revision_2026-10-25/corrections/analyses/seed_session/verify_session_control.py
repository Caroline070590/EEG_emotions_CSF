"""Independent arithmetic/alignment/invariance checks of saved session controls."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
EMOTIONS = ["anger", "disgust", "fear", "sadness", "neutral", "happiness", "surprise"]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    summary = json.loads((OUT / "SESSION_CONTROL_SUMMARY.json").read_text())
    assert summary["analysis_status"] == "complete"
    d = pd.read_csv(OUT / "aligned_out_of_fold_predictions.csv")
    c = pd.read_csv(OUT / "clips.csv").set_index("video")
    orig = pd.read_csv(HERE.parents[2] / "seed7_replication/cache/features_scalar.csv")
    pd.testing.assert_frame_equal(d[["subject", "video", "clip", "session", "emotion", "source"]],
                                 orig[["subject", "video", "clip", "session", "emotion", "source"]])
    assert len(d) == 1600 and not d.duplicated(["subject", "video"]).any()
    y = d.label.to_numpy()
    manual_score = np.mean([np.mean(d.loc[y == i, "lda_prediction"].values == i) for i in range(7)])
    assert np.isclose(manual_score, summary["observed_balanced_accuracy"], atol=1e-15, rtol=0)
    manual_frequency = np.empty((len(d), 7))
    manual_ties = np.empty_like(manual_frequency)
    manual_balanced_ties = np.empty_like(manual_frequency)
    for source in c.source.unique():
        test = d.source.eq(source).to_numpy()
        condition_totals = np.bincount(y[~test], minlength=7)
        for session in range(1, 5):
            tr = ~test & d.session.eq(session).to_numpy()
            te = test & d.session.eq(session).to_numpy()
            freq = np.array([np.count_nonzero(y[tr] == i) for i in range(7)], dtype=float)
            manual_frequency[te] = freq / freq.sum()
            winner = freq == freq.max()
            manual_ties[te] = winner / winner.sum()
            adjusted = freq / condition_totals
            winner_balanced = np.isclose(adjusted, adjusted.max(), rtol=0, atol=1e-14)
            manual_balanced_ties[te] = winner_balanced / winner_balanced.sum()
    for i, emotion in enumerate(EMOTIONS):
        assert np.allclose(manual_frequency[:, i], d[f"session_frequency_probability_{emotion}"])
        assert np.allclose(manual_ties[:, i], d[f"session_majority_tie_probability_{emotion}"])
    frequency_score = np.mean([manual_frequency[y == i, i].mean() for i in range(7)])
    tie_score = np.mean([manual_ties[y == i, i].mean() for i in range(7)])
    assert np.isclose(frequency_score, summary["session_frequency_expected_balanced_accuracy"], atol=1e-15, rtol=0)
    assert tie_score == summary["session_majority_tie_averaged_expected_balanced_accuracy"] == 0
    assert np.all(manual_ties[np.arange(len(y)), y] == 0)
    support_summary = json.loads((OUT / "SESSION_SUPPORT_REFERENCE.json").read_text())
    supported = manual_frequency > 0
    assert np.all(supported.sum(axis=1) == 5)
    support_probabilities = supported / supported.sum(axis=1, keepdims=True)
    support_expected = np.mean([support_probabilities[y == i, i].mean() for i in range(7)])
    assert support_expected == support_summary["expected_balanced_accuracy"] == .2
    support_rows = pd.read_csv(OUT / "aligned_session_support_reference.csv")
    pd.testing.assert_frame_equal(support_rows[["subject", "video", "session", "source", "fold", "label", "lda_prediction"]],
                                 d[["subject", "video", "session", "source", "fold", "label", "lda_prediction"]])
    for i in range(7):
        assert np.array_equal(support_probabilities[:, i], support_rows[f"session_support_probability_label_{i}"].values)
    additional_summary = json.loads((OUT / "ADDITIONAL_SESSION_REFERENCES.json").read_text())
    additional_rows = pd.read_csv(OUT / "additional_aligned_session_references.csv")
    pd.testing.assert_frame_equal(additional_rows[["subject", "video", "session", "source", "fold", "label"]],
                                 d[["subject", "video", "session", "source", "fold", "label"]])
    additional_lda_score = np.mean([np.mean(additional_rows.loc[y == i, "session_lda_prediction"].values == i) for i in range(7)])
    assert additional_lda_score == additional_summary["session_one_hot_shrinkage_lda_balanced_accuracy"] == 0
    balanced_tie_score = np.mean([manual_balanced_ties[y == i, i].mean() for i in range(7)])
    assert balanced_tie_score == additional_summary["inverse_training_class_frequency_weighted_session_tie_averaged_expected_balanced_accuracy"]
    for i in range(7):
        assert np.allclose(manual_balanced_ties[:, i], additional_rows[f"class_balanced_session_tie_probability_label_{i}"].values)

    original_labels = np.array([EMOTIONS.index(e) for e in c.emotion])
    draws = np.load(OUT / "restricted_permutation_clip_labels.npy")
    null = np.load(OUT / "restricted_null_balanced_accuracy.npy")
    assert len(draws) == len(null) == summary["completed_permutations"] == 1000
    source_groups = list(c.groupby("source"))
    mixed_sources = [source for source, g in source_groups if g.emotion.nunique() > 1]
    for labels in draws:
        for session, g in c.groupby("session"):
            assert np.array_equal(np.bincount(labels[g.index], minlength=7),
                                  np.bincount(original_labels[g.index], minlength=7))
        for source, g in source_groups:
            if source in mixed_sources:
                assert np.array_equal(labels[g.index], original_labels[g.index])
            else:
                assert len(np.unique(labels[g.index])) == 1
    n_unique_draws = np.unique(draws, axis=0).shape[0]
    assert n_unique_draws == summary["n_unique_draws"]
    unique_allocations = 1
    for _, row in pd.read_csv(OUT / "permutation_strata.csv").iterrows():
        sources = json.loads(row.sources)
        labels = c[c.source.isin(sources)].groupby("source").emotion.first()
        counts = labels.value_counts().values
        unique = math.factorial(len(sources)) // math.prod(math.factorial(int(x)) for x in counts)
        assert unique == int(row.n_unique_label_allocations)
        unique_allocations *= unique
    assert str(unique_allocations) == summary["n_unique_possible_label_allocations"]
    assert summary["restricted_null_exceedances"] == int(np.count_nonzero(null >= manual_score))
    assert summary["restricted_null_p_plus_one"] == (1 + np.count_nonzero(null >= manual_score)) / (1 + len(null))
    assert np.isclose(summary["restricted_null_mean"], null.mean(), atol=1e-15, rtol=0)
    assert np.isclose(summary["restricted_null_95th_percentile"], np.quantile(null, .95), atol=1e-15, rtol=0)
    for file, expected in summary["input_sha256"].items():
        assert digest(Path(file)) == expected
    assert digest(HERE / "run_session_control.py") == summary["script_sha256"]
    exchange = pd.read_csv(OUT / "source_exchangeability.csv")
    fixed_sources = exchange.loc[~exchange.permutable, "source"]
    result = {"status": "passed", "rows_aligned": len(d), "folds": d.fold.nunique(),
              "manual_observed_balanced_accuracy": manual_score,
              "manual_session_frequency_expected_balanced_accuracy": frequency_score,
              "manual_session_majority_tie_expected_balanced_accuracy": tie_score,
              "manual_session_support_expected_balanced_accuracy": support_expected,
              "manual_inverse_class_frequency_session_tie_expected_balanced_accuracy": balanced_tie_score,
              "permutation_maps_verified": len(draws), "unique_maps": n_unique_draws,
              "session_condition_counts_preserved": True,
              "source_excerpt_dependence_preserved": True,
              "fixed_source_count": len(fixed_sources),
              "fixed_clip_count": int(c.source.isin(fixed_sources).sum()),
              "exchangeable_clip_count": int((~c.source.isin(fixed_sources)).sum()),
              "input_hashes_unchanged": True,
              "result_hashes": {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
    (HERE / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "result_hashes"}, indent=2))


if __name__ == "__main__":
    main()
