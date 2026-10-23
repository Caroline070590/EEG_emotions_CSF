"""Sensitivity of leave-one-video-out decoding to how the within-participant standardization is fitted.

The primary decoding (fig3_decoding_unseen_videos.py) z-scores features within participant using all of that
participant's videos, including the held-out video (transductive; no labels are used). Here the
standardization is refitted in every fold on the participant's training videos only and applied to the
held-out video. Observed balanced accuracy only (band-specific synchronization + power, 160 features).
Output: tables/normalization_sensitivity.csv
"""
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from lib_rev import load, EMOS, BANDS5, TAB

df, arr = load()
X = np.c_[df[[f"{s}_{b}" for s in ("wpli", "oaec") for b in BANDS5]].values, np.hstack([arr[f"logpow_{b}"] for b in BANDS5])]
subj, video = df.subject.values, df.video.values
y = np.array([EMOS.index(e) for e in df.emotion])

def norm(fit):
    Z = np.empty_like(X, float)
    for s in np.unique(subj):
        m = subj == s; f = m & fit
        Z[m] = (X[m] - X[f].mean(0)) / (X[f].std(0) + 1e-12)
    return Z

rows = []; Zall = norm(np.ones(len(X), bool))
for mode in ("all videos (primary, transductive)", "training videos only"):
    pred = np.empty_like(y)
    for v in np.unique(video):
        te = video == v; tr = ~te; Z = Zall if mode.startswith("all") else norm(tr)
        sc = StandardScaler().fit(Z[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(Z[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(Z[te]))
    rows.append({"dataset": "FACED", "features": "Band-specific sync + power", "normalization_fitted_on": mode,
                 "balanced_accuracy": balanced_accuracy_score(y, pred)})
    print(rows[-1])
pd.DataFrame(rows).to_csv(TAB / "normalization_sensitivity.csv", index=False)
