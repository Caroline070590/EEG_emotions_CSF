"""
regions.py — REGION-grouped small-world analysis (faithful port of region.ipynb).

Per trial x region:
  Spearman : region channels -> channel-channel matrix -> metrics
  MTF / QG : each channel -> 1D quantile transition matrix -> metrics,
             then AVERAGE channel-level metrics within the region.
The primary analysis uses fixed q=30. Resumable, parallel driver writes
cache/region_trial_metrics.csv (long form: subject,emotion,region,method,...).
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib.config import CACHE, EMOS, TARGET_DENSITY, N_RANDOMIZATIONS
from lib.representations import (load_raw_trial_csv, parse_subject_trial,
                                 choose_q_campanharo, choose_k_from_autocorrelation,
                                 matrix_spearman_channels, quantile_symbols, _list_trials)
from lib.metrics import metrics_from_matrix

# region map (faithful to region.ipynb, matched to the 30 electrodes present)
REGION_MAP = {
    "frontal":   ["FP1","FP2","F3","F4","F7","F8","FZ"],
    "central":   ["FC1","FC2","FC5","FC6","CZ","C3","C4"],
    "temporal":  ["T7","T8"],
    "parietal":  ["CP1","CP2","CP5","CP6","P3","P4","P7","P8","PZ"],
    "occipital": ["PO3","PO4","O1","O2","OZ"],
}
REGIONS = list(REGION_MAP.keys())
from lib.config import CH
COLIDX = {c: i for i, c in enumerate(CH)}

def _mtf_1d(x, q, lag=1):
    x = np.asarray(x, float); T = len(x); lag = max(1, min(int(lag), T-2))
    A = np.zeros((q, q))
    if np.nanstd(x) == 0: return A
    s = quantile_symbols(x, q)
    for t in range(0, T-lag): A[s[t], s[t+lag]] += 1.0
    np.fill_diagonal(A, 0.0); rs = A.sum(1, keepdims=True)
    A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    A = (A + A.T)/2; np.fill_diagonal(A, 0.0); return A

def _choose_k_1d(x, thr=0.10, max_frac=0.25):
    x = np.asarray(x, float); T = len(x)
    max_lag = min(max(2, int(np.floor(T*max_frac))), T-2)
    if max_lag < 2: return 1
    xx = x - x.mean(); den = np.sum(xx**2)
    if den <= 0: return 1
    acf = np.array([np.sum(xx[:-l]*xx[l:])/den for l in range(1, max_lag+1)])
    below = np.where(np.abs(acf) <= thr)[0]
    return int(below[0]+1) if len(below) else int(np.nanargmin(np.abs(acf))+1)

def _region_metrics_one(path, emo, n_rand):
    X = load_raw_trial_csv(path); subj, tr = parse_subject_trial(path)
    T = X.shape[0]; q = choose_q_campanharo(T)
    rows = []
    for region, chans in REGION_MAP.items():
        idx = [COLIDX[c] for c in chans if c in COLIDX]
        if len(idx) == 0: continue
        base = dict(subject=subj, emotion=emo, trial=Path(path).stem, region=region,
                    n_region_channels=len(idx), q=q)
        sig = X[:, idx]
        # Spearman (region channel graph)
        r = {**base, "method": "Spearman", **metrics_from_matrix(matrix_spearman_channels(sig), n_rand=n_rand)}
        rows.append(r)
        # MTF (per channel, average metrics)
        mt = [metrics_from_matrix(_mtf_1d(X[:, c], q, lag=1), n_rand=n_rand) for c in idx]
        rows.append({**base, "method": "MTF", **{k: np.nanmean([d[k] for d in mt]) for k in mt[0]}})
        # QG (per channel, average metrics; k per channel)
        qg = [metrics_from_matrix(_mtf_1d(X[:, c], q, lag=_choose_k_1d(X[:, c])), n_rand=n_rand) for c in idx]
        rows.append({**base, "method": "QG", **{k: np.nanmean([d[k] for d in qg]) for k in qg[0]}})
    return rows

def compute_region_metrics(n_jobs=-1, n_rand=N_RANDOMIZATIONS, batch=80):
    """Resumable: appends in batches, skips completed trials."""
    f = CACHE / "region_trial_metrics.csv"
    done = set()
    if f.exists():
        d = pd.read_csv(f, usecols=["subject", "trial"]); done = set(zip(d["subject"], d["trial"]))
    items = _list_trials()
    import re
    todo = [(p, emo) for p, _, emo in items
            if (int(re.search(r"Patient[_\-]?(\d+)", p.parent.name, 2).group(1)), p.stem) not in done]
    print(f"region metrics: total {len(items)} | done {len(done)} | todo {len(todo)}", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i+batch]
        out = Parallel(n_jobs=n_jobs)(delayed(_region_metrics_one)(p, emo, n_rand) for p, emo in chunk)
        flat = [r for rows in out for r in rows]
        pd.DataFrame(flat).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"  +{len(chunk)} trials ({i+len(chunk)}/{len(todo)})", flush=True)
    print("DONE", flush=True)
    return pd.read_csv(f)
