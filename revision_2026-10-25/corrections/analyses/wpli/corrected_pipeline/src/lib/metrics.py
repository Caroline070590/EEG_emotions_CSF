"""
metrics.py — faithful port of the small-world network metrics from
Spearman-MTF-QG.ipynb (threshold to fixed density, randomised references,
sigma/gamma/lambda), plus a parallel, checkpointed trial-metrics driver that
writes trial_metrics_{spearman,mtf,qg}.csv.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed

from lib.config import (CACHE, EMOS, VAL, TARGET_DENSITY, N_RANDOMIZATIONS,
                        REWIRES_PER_EDGE)
from lib.representations import (load_raw_trial_csv, parse_subject_trial,
                                 choose_q_campanharo, choose_k_from_autocorrelation,
                                 matrix_spearman_channels, matrix_markov_transition,
                                 _list_trials)

def _prep(A):
    A = np.nan_to_num(np.asarray(A, float)); A = np.abs(A); A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0); return A

def threshold_to_density(A, density=TARGET_DENSITY):
    A = _prep(A); n = A.shape[0]
    if n < 2: return np.zeros_like(A, int)
    tri = np.triu_indices(n, 1); w = A[tri]; valid = np.isfinite(w) & (w > 0)
    if valid.sum() == 0: return np.zeros_like(A, int)
    n_keep = max(1, min(int(np.round(density * (n*(n-1)//2))), int(valid.sum())))
    cutoff = np.sort(w[valid])[-n_keep]
    B = np.zeros_like(A, int); B[(A >= cutoff) & (A > 0)] = 1
    B = np.maximum(B, B.T); np.fill_diagonal(B, 0); return B

def _gcc(G):
    if G.number_of_nodes() <= 1 or nx.is_connected(G): return G
    comps = list(nx.connected_components(G))
    return G.subgraph(max(comps, key=len)).copy() if comps else G

def _rand_refs(G, n_rand, rewires, seed=123):
    if G.number_of_edges() < 2 or G.number_of_nodes() < 4: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); C, L, E = [], [], []
    for _ in range(n_rand):
        Gr = G.copy(); nsw = max(1, rewires * Gr.number_of_edges())
        try:
            nx.double_edge_swap(Gr, nswap=nsw, max_tries=max(100, 20*nsw),
                                seed=int(rng.integers(0, 1_000_000)))
        except Exception:
            continue
        if Gr.number_of_edges() == 0: continue
        gcc = _gcc(Gr)
        try: C.append(nx.average_clustering(Gr))
        except Exception: C.append(np.nan)
        try: L.append(nx.average_shortest_path_length(gcc) if gcc.number_of_nodes() > 1 else np.nan)
        except Exception: L.append(np.nan)
        try: E.append(nx.global_efficiency(Gr))
        except Exception: E.append(np.nan)
    return (np.nanmean(C) if C else np.nan, np.nanmean(L) if L else np.nan,
            np.nanmean(E) if E else np.nan)

def metrics_from_matrix(A, density=TARGET_DENSITY, n_rand=N_RANDOMIZATIONS):
    B = threshold_to_density(A, density); G = nx.from_numpy_array(B)
    ne, nn = G.number_of_edges(), G.number_of_nodes()
    if ne == 0 or nn < 3:
        return {k: np.nan for k in ["transitivity","global_efficiency","char_path_len_gcc",
                "gamma_C_over_Crand","lambda_L_over_Lrand","sigma_small_world"]}
    gcc = _gcc(G)
    C = nx.average_clustering(G); Tt = nx.transitivity(G); E = nx.global_efficiency(G)
    L = nx.average_shortest_path_length(gcc) if gcc.number_of_nodes() > 1 else np.nan
    Cr, Lr, _ = _rand_refs(G, n_rand, REWIRES_PER_EDGE)
    gamma = C/Cr if (Cr and Cr > 0) else np.nan
    lambd = L/Lr if (Lr and Lr > 0) else np.nan
    sigma = gamma/lambd if (np.isfinite(gamma) and np.isfinite(lambd) and lambd > 0) else np.nan
    return {"transitivity": Tt, "global_efficiency": E, "char_path_len_gcc": L,
            "gamma_C_over_Crand": gamma, "lambda_L_over_Lrand": lambd, "sigma_small_world": sigma}

def _metrics_one(path, emo, n_rand):
    X = load_raw_trial_csv(path); subj, tr = parse_subject_trial(path)
    T = X.shape[0]; q = choose_q_campanharo(T); k = choose_k_from_autocorrelation(X)
    base = dict(subject=subj, emotion=emo, trial=Path(path).stem, T=T, q=q, k=k)
    res = {}
    res["Spearman"] = {**base, **metrics_from_matrix(matrix_spearman_channels(X), n_rand=n_rand)}
    res["MTF"] = {**base, **metrics_from_matrix(matrix_markov_transition(X, q=q, lag=1), n_rand=n_rand)}
    res["QG"]  = {**base, **metrics_from_matrix(matrix_markov_transition(X, q=q, lag=k), n_rand=n_rand)}
    return res

def compute_trial_metrics(n_jobs=-1, n_rand=N_RANDOMIZATIONS, force=False):
    """Parallel + checkpointed. Writes cache/trial_metrics_{method}.csv."""
    files = {"spearman": CACHE/"trial_metrics_spearman.csv",
             "mtf": CACHE/"trial_metrics_mtf.csv", "qg": CACHE/"trial_metrics_qg.csv"}
    if all(f.exists() for f in files.values()) and not force:
        print("[metrics] cached"); return {m: pd.read_csv(f) for m, f in files.items()}
    items = _list_trials(); t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_metrics_one)(p, emo, n_rand) for p, _, emo in items)
    for tag, key in [("spearman","Spearman"),("mtf","MTF"),("qg","QG")]:
        pd.DataFrame([r[key] for r in out]).to_csv(files[tag], index=False)
    print(f"[metrics] {len(items)} trials x3 methods in {time.time()-t0:.1f}s")
    return {m: pd.read_csv(f) for m, f in files.items()}
