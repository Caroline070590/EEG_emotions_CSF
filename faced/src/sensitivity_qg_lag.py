"""Sensitivity of the QG results to how the QG lag is estimated (added after an internal review).

The primary QG lag is the first lag at which the autocorrelation of the channel-AVERAGE signal falls to
|ACF| <= 0.10. After common-average referencing, the average of the retained channels mostly reflects
the reference channels that were not retained, so its timescale is hard to interpret. Here the lag is
instead estimated per channel (same rule) and the median over the 30 channels is used. The QG network
(q = 30, 10% density) and its six topology metrics are recomputed with exactly the primary code, and the
48 emotion-versus-neutral tests are rerun (participants and videos random; BH within the 48 tests).

Outputs: tables/qg_lag_sensitivity.csv (one row per metric x contrast, both lag rules) and
cache/qg_lag_alt.csv. usage: sensitivity_qg_lag.py [N_JOBS]
"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
spec = importlib.util.spec_from_file_location("feat", HERE / "src" / "01_trial_features.py")
feat = importlib.util.module_from_spec(spec); spec.loader.exec_module(feat)
from lib_rev import load, contrast_vs_neutral, bh, NONNEU, TOPO, CACHE, TAB

def channel_lag(x, thr=0.10, max_frac=0.25):
    return feat.qg_lag(x[None, :], thr, max_frac)

def one(i, mm_path):
    X = np.load(mm_path, mmap_mode="r")[i].astype(np.float64)
    lags = [channel_lag(x) for x in X]
    k = int(np.median(lags))
    A = feat.transition_matrix(X, feat.PRIMARY_Q, k)
    m = feat.topo_metrics(A, feat.PRIMARY_D, feat.N_RAND_PRIMARY)
    return i, k, float(np.min(lags)), float(np.max(lags)), m

if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    df, _ = load()
    mm = CACHE / "eeg_30s.npy"
    out = Parallel(n_jobs=n_jobs, verbose=5, batch_size=4)(delayed(one)(int(i), mm) for i in range(len(df)))
    out.sort(key=lambda t: t[0])
    alt = pd.DataFrame({"qg_lag_alt": [o[1] for o in out], "lag_min": [o[2] for o in out], "lag_max": [o[3] for o in out]})
    for j, mname in enumerate(TOPO):
        alt[f"QGalt_q30_d10_{mname}"] = [o[4][j] for o in out]
    alt.to_csv(CACHE / "qg_lag_alt.csv", index=False)
    df = pd.concat([df.reset_index(drop=True), alt], axis=1)

    rows = []
    for mname in TOPO:
        for rule, col in (("channel-average (primary)", f"QG_q30_d10_{mname}"), ("median per-channel (sensitivity)", f"QGalt_q30_d10_{mname}")):
            y = df[col].to_numpy(float)
            for e in NONNEU:
                rows.append({"metric": mname, "lag_rule": rule, **contrast_vs_neutral(y, df, e)})
    R = pd.DataFrame(rows)
    for rule in R.lag_rule.unique():
        m = R.lag_rule == rule
        R.loc[m, "q_minF"] = bh(R.loc[m, "p_minF"]); R.loc[m, "q_participant_only"] = bh(R.loc[m, "p1"])
    R.to_csv(TAB / "qg_lag_sensitivity.csv", index=False)
    a, b = (R[R.lag_rule == r].reset_index(drop=True) for r in R.lag_rule.unique())
    print("lag (samples): primary median", int(df.qg_lag.median()), "| alternative median", int(df.qg_lag_alt.median()),
          "IQR", df.qg_lag_alt.quantile([.25, .75]).tolist())
    print("trial-level r between lag rules:", round(np.corrcoef(df.qg_lag, df.qg_lag_alt)[0, 1], 3))
    print("d_z agreement across the 48 contrasts: r =", round(np.corrcoef(a.dz, b.dz)[0, 1], 3),
          "| same sign", round(float((np.sign(a.dz) == np.sign(b.dz)).mean()), 3))
    print(R.groupby("lag_rule").apply(lambda g: pd.Series({"q<.05 videos fixed": int((g.q_participant_only < .05).sum()),
                                                          "q<.05 videos random": int((g.q_minF < .05).sum())})).to_string())
