"""Retain two exploratory session-only reference checks, without formal tests.

Both use the exact source-held-out folds. One-hot session LDA matches the EEG
classifier's algorithm. Inverse-training-class-frequency weighting instead
optimizes training balanced accuracy under the empirical session table; maximum
ties are averaged explicitly. These checks do not replace the restricted null.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent / "results"
d = pd.read_csv(OUT / "aligned_out_of_fold_predictions.csv")
y = d.label.to_numpy()
X = pd.get_dummies(d.session, dtype=float).to_numpy()
lda_prediction = np.full(len(d), -1)
balanced_prediction = np.full(len(d), -1)
balanced_ties = np.zeros((len(d), 7))
with threadpool_limits(limits=1):
    for source in d.source.unique():
        test = d.source.eq(source).to_numpy()
        scaler = StandardScaler().fit(X[~test])
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(scaler.transform(X[~test]), y[~test])
        lda_prediction[test] = model.predict(scaler.transform(X[test]))
        training_class_counts = np.bincount(y[~test], minlength=7)
        assert np.all(training_class_counts > 0)
        for session in range(1, 5):
            tr = ~test & d.session.eq(session).to_numpy()
            te = test & d.session.eq(session).to_numpy()
            counts = np.bincount(y[tr], minlength=7)
            # Equivalent to P(class | session) / P(class), up to a common factor.
            score = counts / training_class_counts
            maximum = np.isclose(score, score.max(), rtol=0, atol=1e-14)
            balanced_ties[te] = maximum / maximum.sum()
            balanced_prediction[te] = score.argmax()
manual_ba = lambda predictions: float(np.mean([np.mean(predictions[y == i] == i) for i in range(7)]))
expected_balanced = float(np.mean([balanced_ties[y == i, i].mean() for i in range(7)]))
rows = d[["subject", "video", "session", "source", "fold", "label"]].copy()
rows["session_lda_prediction"] = lda_prediction
rows["class_balanced_session_first_tie_prediction"] = balanced_prediction
rows["class_balanced_session_tie_expected_correct"] = balanced_ties[np.arange(len(d)), y]
for i in range(7):
    rows[f"class_balanced_session_tie_probability_label_{i}"] = balanced_ties[:, i]
rows.to_csv(OUT / "additional_aligned_session_references.csv", index=False)
summary = {
    "post_audit_exploratory_reference_checks": True,
    "session_one_hot_shrinkage_lda_balanced_accuracy": manual_ba(lda_prediction),
    "inverse_training_class_frequency_weighted_session_first_tie_balanced_accuracy": manual_ba(balanced_prediction),
    "inverse_training_class_frequency_weighted_session_tie_averaged_expected_balanced_accuracy": expected_balanced,
    "class_balanced_rule": "Argmax of session-by-condition training count divided by total condition training count; ties averaged uniformly.",
    "scope": "Same 54 source-held-out folds, training labels only; no EEG features; no p-values or superiority claims. These do not replace the restricted session/source null.",
}
(OUT / "ADDITIONAL_SESSION_REFERENCES.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
