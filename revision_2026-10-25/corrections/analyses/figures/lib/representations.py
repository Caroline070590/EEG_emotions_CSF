"""
representations.py — faithful port of the Spearman / MTF / QG network builders
from Spearman-MTF-QG.ipynb, parallelised.

  Spearman : 30 EEG channels, |Spearman corr|, diag 0                 (30x30)
  MTF      : quantile-state Markov transition, lag = 1                 (qxq)
  QG       : quantile graph, lag = k (first |ACF| <= 10%)             (qxq)
  q = 30 for the primary manuscript analysis, independently of segment length.

Two roles:
  * trial METRICS (Part 1): fixed q=30.
  * REPRESENTATION matrices for the embedding (Part 3): fixed q=30.
"""
from pathlib import Path
import re, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from lib.config import (CH, EMOS, VAL, DATA_ROOT, CACHE, DROP_COLS_LIKE,
                        Q_FIXED, Q_MIN, Q_MAX, ACF_THRESHOLD_FOR_K, MAX_K_FRACTION,
                        Q_EMBED, REPR_N)

# ---------------------------------------------------------------- IO
def load_raw_trial_csv(path):
    """time x channels, 30 real electrodes (index/time columns excluded)."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns if any(k in str(c).lower() for k in DROP_COLS_LIKE)]
    X = df.drop(columns=drop, errors="ignore").apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all").dropna(axis=0, how="all").to_numpy(float)
    if X.shape[0] < X.shape[1]:
        X = X.T
    if np.isnan(X).any():
        med = np.nanmedian(X, axis=0); idx = np.where(np.isnan(X)); X[idx] = np.take(med, idx[1])
    return X

def parse_subject_trial(path):
    pf, tf = Path(path).parent.name, Path(path).stem
    s = re.search(r"Patient[_\-]?(\d+)", pf, re.I); subj = int(s.group(1)) if s else pf
    t = re.search(r"trial[_\-]?(\d+)", tf, re.I); tr = int(t.group(1)) if t else tf
    return subj, tr

# ---------------------------------------------------------------- q, k, symbols
def choose_q_campanharo(T):
    """Compatibility name retained for older callers; primary q is fixed."""
    del T
    return Q_FIXED

def autocorr_1d(x, max_lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < max_lag + 5: return np.full(max_lag, np.nan)
    x = x - x.mean(); denom = np.sum(x**2)
    if denom <= 0: return np.full(max_lag, np.nan)
    return np.array([np.sum(x[:-l] * x[l:]) / denom for l in range(1, max_lag+1)])

def choose_k_from_autocorrelation(X, threshold=ACF_THRESHOLD_FOR_K):
    x = np.nanmean(X, axis=1); T = len(x)
    max_lag = min(max(2, int(np.floor(T * MAX_K_FRACTION))), T - 2)
    if max_lag < 2: return 1
    acf = autocorr_1d(x, max_lag); valid = np.isfinite(acf)
    if valid.sum() == 0: return 1
    below = np.where((np.abs(acf) <= threshold) & valid)[0]
    return int(below[0] + 1) if len(below) else int(np.nanargmin(np.abs(acf)) + 1)

def quantile_symbols(x, q):
    x = np.asarray(x, float)
    if len(x) == 0 or np.nanstd(x) == 0: return np.zeros(len(x), int)
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, q+1)))
    if len(edges) <= 2: return np.zeros(len(x), int)
    sym = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges)-2)
    if len(edges)-1 < q: sym = np.clip(sym, 0, q-1)
    return sym.astype(int)

# ---------------------------------------------------------------- matrices
def matrix_spearman_channels(X):
    n = X.shape[1]
    if n < 2: return np.zeros((n, n))
    corr, _ = spearmanr(X, axis=0)
    A = np.asarray(corr, float) if not np.isscalar(corr) else np.zeros((n, n))
    if A.shape != (n, n): A = np.corrcoef(X.T)
    A = np.nan_to_num(np.abs(A)); np.fill_diagonal(A, 0.0)
    return A

def matrix_markov_transition(X, q, lag=1):
    T, nch = X.shape
    lag = max(1, min(int(lag), T-2))
    A = np.zeros((q, q))
    for ch in range(nch):
        x = X[:, ch]
        if np.nanstd(x) == 0: continue
        s = np.clip(quantile_symbols(x, q), 0, q-1)
        for t in range(0, T - lag):
            A[s[t], s[t+lag]] += 1.0
    np.fill_diagonal(A, 0.0)
    rs = A.sum(1, keepdims=True)
    A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    A = (A + A.T) / 2.0; np.fill_diagonal(A, 0.0)
    return A

def matrix_quantile_graph(X, q, k):
    return matrix_markov_transition(X, q=q, lag=k)

# ---------------------------------------------------------------- triu helpers
def triu_idx(n): return np.triu_indices(n, 1)
def vec_to_mat(v, n):
    M = np.zeros((n, n)); M[triu_idx(n)] = v; M += M.T; np.fill_diagonal(M, 1.0); return M

# ---------------------------------------------------------------- representation build (Part 3)
def _repr_one(path, tag):
    X = load_raw_trial_csv(path)
    if X.shape[0] < 20: return None
    if tag == "spearman":
        A = matrix_spearman_channels(X)
    elif tag == "mtf":
        A = matrix_markov_transition(X, q=Q_EMBED, lag=1)
    elif tag == "qg":
        k = choose_k_from_autocorrelation(X)
        A = matrix_markov_transition(X, q=Q_EMBED, lag=k)
    else:
        raise ValueError(tag)
    n = A.shape[0]
    return A[triu_idx(n)].astype(np.float32)

def _list_trials():
    items = []
    for emo in EMOS:
        for pat in sorted(p for p in (DATA_ROOT / emo).iterdir() if p.is_dir()):
            for tr in sorted(pat.glob("trial_*.csv")):
                items.append((tr, pat.name, emo))
    return items

def build_representation(tag="spearman", n_jobs=-1, force=False):
    f_feat = CACHE / f"repr_{tag}_triu.npy"; f_meta = CACHE / f"repr_{tag}_meta.csv"
    if f_feat.exists() and f_meta.exists() and not force:
        return np.load(f_feat), pd.read_csv(f_meta)
    items = _list_trials(); t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_repr_one)(p, tag) for p, _, _ in items)
    rows, mats = [], []
    for (p, subj, emo), v in zip(items, out):
        if v is None: continue
        rows.append({"subject": subj, "emotion": emo, "trial": Path(p).stem}); mats.append(v)
    feat = np.vstack(mats); meta = pd.DataFrame(rows)
    np.save(f_feat, feat); meta.to_csv(f_meta, index=False)
    print(f"[repr {tag}] {len(meta)} trials in {time.time()-t0:.1f}s -> {feat.shape}")
    return feat, meta

def aggregate(feat, meta):
    meta = meta.copy(); meta["idx"] = np.arange(len(meta))
    rows, agg = [], []
    for (subj, emo), g in meta.groupby(["subject", "emotion"]):
        agg.append(feat[g["idx"].values].mean(0)); rows.append({"subject": subj, "emotion": emo, "valence": VAL[emo]})
    return np.vstack(agg), pd.DataFrame(rows)
