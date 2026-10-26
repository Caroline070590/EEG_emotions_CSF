"""Additional training-only session-support probability reference.

The empirical-frequency majority rule is a poor diagnostic when held-source
folds selectively reduce a test condition's training count. A second, explicit
session-only stochastic reference draws uniformly among conditions present in
the training observations of the test session. Its expected balanced accuracy
is calculated analytically; it is not empirical classifier accuracy. Neither
reference substitutes for the session/source-preserving permutation sensitivity.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
d = pd.read_csv(OUT / "aligned_out_of_fold_predictions.csv")
probabilities = np.full((len(d), 7), np.nan)
y = d.label.to_numpy()
counts = []
for source in d.source.unique():
    held = d.source.eq(source).to_numpy()
    for session in range(1, 5):
        train = ~held & d.session.eq(session).to_numpy()
        test = held & d.session.eq(session).to_numpy()
        support = np.bincount(y[train], minlength=7) > 0
        probabilities[test] = support / support.sum()
        counts.append(int(support.sum()))
assert np.isfinite(probabilities).all()
assert set(counts) == {5}
expected = float(np.mean([probabilities[y == i, i].mean() for i in range(7)]))
assert np.isclose(expected, .2)
true_probability = probabilities[np.arange(len(d)), y]
assert np.allclose(true_probability, .2)
aligned = d[["subject", "video", "session", "source", "fold", "label", "lda_prediction"]].copy()
aligned["session_support_expected_correct"] = true_probability
for i in range(7):
    aligned[f"session_support_probability_label_{i}"] = probabilities[:, i]
aligned.to_csv(OUT / "aligned_session_support_reference.csv", index=False)
rows = []
for subject, frame in d.groupby("subject"):
    ba = float(np.mean([np.mean(frame.loc[frame.label == i, "lda_prediction"].values == i) for i in range(7)]))
    rows.append({"subject": int(subject), "lda_balanced_accuracy": ba,
                 "session_support_expected_balanced_accuracy": expected,
                 "paired_difference": ba - expected})
table = pd.DataFrame(rows)
table.to_csv(OUT / "session_support_paired_scores.csv", index=False)
rng = np.random.default_rng(20261026)
diffs = table.paired_difference.to_numpy()
ci = np.quantile(diffs[rng.integers(0, 20, size=(10000, 20))].mean(1), [.025, .975])
summary = {"status": "complete", "post_audit_sensitivity": True,
           "description": "Uniform draws among the conditions observed in the test session's training data, using the same 54 source-held-out folds.",
           "training_supported_conditions_per_session_in_every_fold": 5,
           "expected_balanced_accuracy": expected,
           "lda_balanced_accuracy": float(table.lda_balanced_accuracy.mean()),
           "paired_difference": float(table.paired_difference.mean()),
           "paired_difference_descriptive_participant_bootstrap_95": ci.tolist(),
           "interpretation": "Analytically expected accuracy of a stochastic session-only reference, not observed deterministic classification accuracy. Participant interval conditions on fitted OOF predictions and sampled sources; not a model-refitting or stimulus-generalization interval.",
           "null_scope": "Restricted session/source-label permutation results are reported separately and are needed to assess session-conditional label association."}
(OUT / "SESSION_SUPPORT_REFERENCE.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
