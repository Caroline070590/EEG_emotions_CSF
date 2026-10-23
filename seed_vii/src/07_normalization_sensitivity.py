"""SEED-VII: sensitivity of leave-one-source-out decoding to how the within-participant standardization is
fitted (all clips of the participant, as in 05_decoding_unseen_sources.py, versus the participant's training clips only,
refitted in every fold). Observed balanced accuracy only. Output: tables/normalization_sensitivity.csv"""
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from lib_rev import load, CACHE, EMOS, BANDS5, TAB

df, arr = load(); rec = np.load(CACHE / "recurrence_features.npz")
SETS = {"Band-specific sync + power": np.c_[df[[f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS5]].values, np.hstack([arr[f"logpow_{b}"] for b in BANDS5])],
        "RQA + JRS": np.c_[rec["rqa"].reshape(len(df), -1), rec["jrs"]], "Spearman edges (signed)": arr["rho_signed"]}
subj, video = df.subject.values, df.video.values
clips = df.groupby("video").agg(emotion=("emotion", "first"), source=("source", "first")).reindex(range(80))
y = np.array([EMOS.index(clips.emotion[v]) for v in video]); g = clips.source.values[video]
rows = []
for name, X in SETS.items():
    def norm(fit):
        Z = np.empty_like(X, float)
        for s in np.unique(subj):
            m = subj == s; f = m & fit; Z[m] = (X[m] - X[f].mean(0)) / (X[f].std(0) + 1e-12)
        return Z
    Zall = norm(np.ones(len(X), bool))
    for mode in ("all clips (primary, transductive)", "training clips only"):
        pred = np.empty_like(y)
        for k in np.unique(g):
            te = g == k; tr = ~te; Z = Zall if mode.startswith("all") else norm(tr)
            sc = StandardScaler().fit(Z[tr]); clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(Z[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(Z[te]))
        rows.append({"dataset": "SEED-VII", "folds": "leave-one-source-out", "features": name, "normalization_fitted_on": mode,
                     "balanced_accuracy": balanced_accuracy_score(y, pred)}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(TAB / "normalization_sensitivity.csv", index=False)
