"""Reproduce observed LDA and selected/full restricted-null scores from snapshots.

No external EEG cache is required. With --all, refits all 1,000 saved restricted
label assignments; by default, checks assignments 0, 1, and 999. The normalized
matrix is independently regenerated from raw feature vectors before fitting.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    data = np.load(OUT / "primary_feature_snapshot.npz")
    frame = pd.read_csv(OUT / "aligned_out_of_fold_predictions.csv")
    raw = data["raw"]
    assert np.array_equal(data["subject"], frame.subject.values)
    assert np.array_equal(data["video"], frame.video.values)
    X = np.empty_like(raw)
    for subject in np.unique(data["subject"]):
        idx = data["subject"] == subject
        X[idx] = (raw[idx] - raw[idx].mean(axis=0)) / (raw[idx].std(axis=0, ddof=1) + 1e-12)
    assert np.allclose(X, data["normalized"], atol=1e-12, rtol=1e-12)
    # Use exact stored pandas-normalized floating values after verifying formula.
    X = data["normalized"]
    permutations = np.load(OUT / "restricted_permutation_clip_labels.npy")
    saved = np.load(OUT / "restricted_null_balanced_accuracy.npy")
    indices = list(range(len(permutations))) if args.all else [0, 1, len(permutations) - 1]
    results = []
    for index in [-1] + indices:
        labels = data["labels"] if index == -1 else permutations[index][data["video"]]
        prediction = np.full(len(frame), -1, dtype=int)
        with threadpool_limits(limits=1):
            for source in sorted(frame.source.unique()):
                te = frame.source.eq(source).values
                sc = StandardScaler().fit(X[~te])
                model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                model.fit(sc.transform(X[~te]), labels[~te])
                prediction[te] = model.predict(sc.transform(X[te]))
        # Arithmetic implementation independent of sklearn's scoring helper.
        score = float(np.mean([np.mean(prediction[labels == c] == c) for c in range(7)]))
        expected = (float(np.mean([np.mean(frame.loc[frame.label == c, "lda_prediction"].values == c)
                                   for c in range(7)])) if index == -1 else float(saved[index]))
        assert np.isclose(score, expected, atol=1e-14, rtol=0), (index, score, expected)
        if index == -1:
            assert np.array_equal(prediction, frame.lda_prediction.values)
        results.append({"permutation": index, "balanced_accuracy": score, "expected": expected})
        print(results[-1], flush=True)
    result = {"status": "passed", "original_out_of_fold_predictions_reproduced_exactly": True,
              "participant_normalization_reproduced_from_raw_features": True,
              "restricted_assignments_refitted": indices, "scores": results,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / ("FULL_SNAPSHOT_REFIT_VERIFICATION.json" if args.all else "SNAPSHOT_REFIT_VERIFICATION.json")).write_text(
        json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
