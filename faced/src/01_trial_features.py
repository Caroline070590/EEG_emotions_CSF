"""Per-trial features on the complete 30-s FACED trials, for the revision analyses.

Graph construction is ported unchanged from the manuscript code (lib/representations.py,
lib/metrics.py): quantile symbols per channel, pooled transitions, self-loops removed,
row-normalised, symmetrised; QG lag = first |ACF| <= 0.10 (max lag 25% of T); fixed-density
binary thresholding; gamma/lambda/sigma against degree-preserving double-edge-swap references.

Analysis choices:
  * full 30-s trials (T = 7500) instead of 250-sample segments
  * standard (amplitude-weighted) wPLI, plus orthogonalised AEC
  * synchronization in canonical bands, not only 1-45 Hz broadband
  * signed Spearman correlations stored alongside |rho|
  * q x density sweep for MTF/QG, density sweep for Spearman
  * split-half and multivariate phase-randomised surrogate versions for the coupling null
  * per-channel band power (incl. 30-45 Hz) for the EMG analysis
  * persistent homology on a proper metric (signed correlation distance)

Output: cache/features_scalar.csv (one row per trial) and cache/features_arrays.npz
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "cache"
FS, T, C = 250, 7500, 30
IU = np.triu_indices(C, 1)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
         "gamma": (30, 45), "broad": (1, 45), "broad4": (4, 45)}
PAIR_BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
DENS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
QS = [10, 20, 25, 30, 35, 40]
PRIMARY_Q, PRIMARY_D = 30, 0.10
N_RAND_PRIMARY, N_RAND_SWEEP, REWIRES = 20, 5, 5
TOPO = ["transitivity", "global_efficiency", "char_path_len_gcc",
        "gamma_C_over_Crand", "lambda_L_over_Lrand", "sigma_small_world"]
SOS = {b: butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos") for b, (lo, hi) in BANDS.items()}


# ------------------------------------------------------------------ graph metrics (ported)
def threshold_to_density(A, density):
    A = np.nan_to_num(np.abs(A)); A = (A + A.T) / 2; np.fill_diagonal(A, 0.0)
    n = A.shape[0]; tri = np.triu_indices(n, 1); w = A[tri]; valid = w > 0
    if valid.sum() == 0: return np.zeros_like(A, int)
    n_keep = max(1, min(int(np.round(density * (n * (n - 1) // 2))), int(valid.sum())))
    cutoff = np.sort(w[valid])[-n_keep]
    B = ((A >= cutoff) & (A > 0)).astype(int); B = np.maximum(B, B.T); np.fill_diagonal(B, 0)
    return B

def _gcc(G):
    if G.number_of_nodes() <= 1 or nx.is_connected(G): return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

def _rand_refs(G, n_rand, seed):
    if G.number_of_edges() < 2 or G.number_of_nodes() < 4: return np.nan, np.nan
    rng = np.random.default_rng(seed); Cs, Ls = [], []
    for _ in range(n_rand):
        Gr = G.copy(); nsw = max(1, REWIRES * Gr.number_of_edges())
        try:
            nx.double_edge_swap(Gr, nswap=nsw, max_tries=max(100, 20 * nsw), seed=int(rng.integers(0, 1_000_000)))
        except Exception:
            continue
        gcc = _gcc(Gr)
        Cs.append(nx.average_clustering(Gr))
        Ls.append(nx.average_shortest_path_length(gcc) if gcc.number_of_nodes() > 1 else np.nan)
    return (np.nanmean(Cs) if Cs else np.nan), (np.nanmean(Ls) if Ls else np.nan)

def topo_metrics(A, density, n_rand, seed=123):
    G = nx.from_numpy_array(threshold_to_density(A, density))
    if G.number_of_edges() == 0: return [np.nan] * 6
    gcc = _gcc(G)
    Cc, Tt, E = nx.average_clustering(G), nx.transitivity(G), nx.global_efficiency(G)
    L = nx.average_shortest_path_length(gcc) if gcc.number_of_nodes() > 1 else np.nan
    Cr, Lr = _rand_refs(G, n_rand, seed)
    gam = Cc / Cr if Cr and Cr > 0 else np.nan
    lam = L / Lr if Lr and Lr > 0 else np.nan
    sig = gam / lam if np.isfinite(gam) and np.isfinite(lam) and lam > 0 else np.nan
    return [Tt, E, L, gam, lam, sig]


# ------------------------------------------------------------------ representations (ported)
def quantile_symbols(x, q):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    if len(edges) <= 2: return np.zeros(len(x), int)
    return np.clip(np.digitize(x, edges[1:-1], right=False), 0, q - 1)

def transition_matrix(X, q, lag):
    """X: C x T. Pooled over channels, self-loops removed, row-normalised, symmetrised."""
    A = np.zeros(q * q)
    for x in X:
        if np.std(x) == 0: continue
        s = quantile_symbols(x, q)
        A += np.bincount(s[:-lag] * q + s[lag:], minlength=q * q)
    A = A.reshape(q, q); np.fill_diagonal(A, 0.0)
    rs = A.sum(1, keepdims=True); A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    A = (A + A.T) / 2; np.fill_diagonal(A, 0.0)
    return A

def qg_lag(X, thr=0.10, max_frac=0.25):
    x = X.mean(0).astype(float); x = x - x.mean(); n = len(x)
    max_lag = min(max(2, int(np.floor(n * max_frac))), n - 2)
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    acf = ac[1:] / ac[0]
    below = np.where(np.abs(acf) <= thr)[0]
    return int(below[0] + 1) if len(below) else int(np.argmin(np.abs(acf)) + 1)

def spearman_signed(X):
    R = rankdata(X, axis=1)
    return np.corrcoef(R)


# ------------------------------------------------------------------ synchronization
def _pairwise_sync(Z):
    """Z: T x C analytic signal. Returns dict of pair vectors."""
    Zc, Zd = Z[:, IU[0]], Z[:, IU[1]]
    ac, ad = np.abs(Zc), np.abs(Zd)
    cs = Zc * np.conj(Zd)
    ph = cs / np.maximum(np.abs(cs), 1e-20)
    plv = np.abs(ph.mean(0))
    Im = cs.imag
    pli = np.abs(np.sign(Im).mean(0))
    den = np.abs(Im).mean(0)
    wpli = np.divide(np.abs(Im.mean(0)), den, out=np.zeros_like(den), where=den > 0)
    def corr(a, b):
        a = a - a.mean(0); b = b - b.mean(0)
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-30)
    aec = corr(ac, ad)
    # pairwise orthogonalisation (Hipp et al., 2012), symmetrised
    yo = np.abs((Zd * np.conj(Zc) / np.maximum(ac, 1e-20)).imag)
    xo = np.abs((Zc * np.conj(Zd) / np.maximum(ad, 1e-20)).imag)
    oaec = 0.5 * (corr(ac, yo) + corr(ad, xo))
    return {"plv": plv, "pli": pli, "wpli": wpli, "aec": aec, "oaec": oaec}

def _xcorr(Xf):
    """Xf: C x T filtered. Max |normalised cross-correlation| and |lag| in ms, per pair."""
    n = Xf.shape[1]
    Xn = (Xf - Xf.mean(1, keepdims=True)) / (Xf.std(1, keepdims=True) + 1e-12)
    F = np.fft.rfft(Xn, 2 * n, axis=1)
    cc = np.fft.irfft(F[IU[0]] * np.conj(F[IU[1]]), 2 * n, axis=1) / n
    cc = np.concatenate([cc[:, -(n - 1):], cc[:, :n]], axis=1)
    k = np.argmax(np.abs(cc), axis=1)
    return np.abs(cc[np.arange(len(k)), k]), np.abs(k - (n - 1)) / FS * 1000

def broadband_sync_summary(X):
    """Six manuscript measures, 1-45 Hz, mean over pairs (used for split-half / surrogate)."""
    Xf = sosfiltfilt(SOS["broad"], X, axis=1)
    p = _pairwise_sync(hilbert(Xf, axis=1).T)
    mx, lag = _xcorr(Xf)
    return {"plv": p["plv"].mean(), "pli": p["pli"].mean(), "wpli": p["wpli"].mean(),
            "aec": p["aec"].mean(), "maxxc": mx.mean(), "lag_ms": lag.mean()}


# ------------------------------------------------------------------ surrogate
def phase_randomised(X, rng):
    """Multivariate phase randomisation: identical random phases across channels preserve
    every auto- and cross-spectrum (all linear inter-channel structure)."""
    F = np.fft.rfft(X, axis=1)
    phi = rng.uniform(0, 2 * np.pi, F.shape[1]); phi[0] = 0
    if X.shape[1] % 2 == 0: phi[-1] = 0
    return np.fft.irfft(F * np.exp(1j * phi)[None, :], n=X.shape[1], axis=1)


# ------------------------------------------------------------------ persistent homology
def ph_features(D):
    import gudhi
    st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=np.inf).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    out = {}
    for dim in (0, 1):
        iv = np.array([d for d in st.persistence_intervals_in_dimension(dim) if np.isfinite(d[1])]).reshape(-1, 2)
        life = iv[:, 1] - iv[:, 0] if len(iv) else np.array([])
        life = life[life > 0]
        tot = life.sum()
        p = life / tot if tot > 0 else np.array([])
        out[f"H{dim}_count"] = len(life)
        out[f"H{dim}_total"] = tot
        out[f"H{dim}_max"] = life.max() if len(life) else 0.0
        out[f"H{dim}_entropy"] = float(-(p * np.log(p)).sum()) if len(p) else 0.0
    return out

def triangle_violation_rate(D, rng, n=4000):
    i, j, k = rng.integers(0, len(D), (3, n))
    ok = (i != j) & (j != k) & (i != k)
    return float(np.mean(D[i, k][ok] > D[i, j][ok] + D[j, k][ok] + 1e-12))


# ------------------------------------------------------------------ per-trial driver
def one_trial(idx, mm_path):
    X = np.load(mm_path, mmap_mode="r")[idx].astype(np.float64)
    rng = np.random.default_rng(1000 + idx)
    row, arr = {"idx": idx}, {}

    # band power (Welch, 2-s windows) per channel
    f, P = welch(X, fs=FS, nperseg=500, axis=1)
    for b, (lo, hi) in BANDS.items():
        if b == "broad4": continue
        arr[f"logpow_{b}"] = np.log10(P[:, (f >= lo) & (f < hi)].mean(1)).astype(np.float32)

    # synchronization per band
    for b in BANDS:
        Xf = sosfiltfilt(SOS[b], X, axis=1)
        p = _pairwise_sync(hilbert(Xf, axis=1).T)
        for m, v in p.items():
            row[f"{m}_{b}"] = v.mean()
        if b in PAIR_BANDS:
            for m in ("wpli", "aec", "oaec", "plv"):
                arr[f"pair_{m}_{b}"] = p[m].astype(np.float32)
        if b == "broad":
            mx, lag = _xcorr(Xf)
            row["maxxc_broad"], row["lag_ms_broad"] = mx.mean(), lag.mean()

    # Spearman (signed and absolute)
    R = spearman_signed(X)
    arr["rho_signed"] = R[IU].astype(np.float32)
    row["rho_neg_fraction"] = float((R[IU] < 0).mean())
    row["rho_mean_signed"] = float(R[IU].mean())
    for d in DENS:
        nr = N_RAND_PRIMARY if d == PRIMARY_D else N_RAND_SWEEP
        for m, v in zip(TOPO, topo_metrics(np.abs(R), d, nr)):
            row[f"Spearman_q30_d{int(d*100):02d}_{m}"] = v
    Rpos = np.clip(R, 0, None)
    for m, v in zip(TOPO, topo_metrics(Rpos, PRIMARY_D, N_RAND_PRIMARY)):
        row[f"SpearmanPos_q30_d10_{m}"] = v

    # MTF / QG, q x density sweep
    k = qg_lag(X); row["qg_lag"] = k
    for q in QS:
        for name, lag in (("MTF", 1), ("QG", k)):
            A = transition_matrix(X, q, lag)
            for d in DENS:
                nr = N_RAND_PRIMARY if (q == PRIMARY_Q and d == PRIMARY_D) else N_RAND_SWEEP
                for m, v in zip(TOPO, topo_metrics(A, d, nr)):
                    row[f"{name}_q{q}_d{int(d*100):02d}_{m}"] = v
            if q == PRIMARY_Q:
                arr[f"{name.lower()}_q30"] = A[np.triu_indices(q, 1)].astype(np.float32)

    # coupling null 1: independent halves
    for h, sl in (("h1", slice(0, T // 2)), ("h2", slice(T // 2, T))):
        Xh = X[:, sl]
        for m, v in broadband_sync_summary(Xh).items(): row[f"{h}_{m}"] = v
        for m, v in zip(TOPO, topo_metrics(np.abs(spearman_signed(Xh)), PRIMARY_D, N_RAND_PRIMARY)):
            row[f"{h}_Spearman_{m}"] = v
    # coupling null 2: multivariate phase-randomised surrogate
    Xs = phase_randomised(X, rng)
    for m, v in broadband_sync_summary(Xs).items(): row[f"surr_{m}"] = v
    for m, v in zip(TOPO, topo_metrics(np.abs(spearman_signed(Xs)), PRIMARY_D, N_RAND_PRIMARY)):
        row[f"surr_Spearman_{m}"] = v

    # persistent homology: signed correlation distance (a metric) and the original |rho| version
    Ds = np.sqrt(np.maximum(0, 2 * (1 - R))); np.fill_diagonal(Ds, 0)
    Da = np.sqrt(np.maximum(0, 2 * (1 - np.abs(R)))); np.fill_diagonal(Da, 0)
    for m, v in ph_features(Ds).items(): row[f"PHsigned_{m}"] = v
    for m, v in ph_features(Da).items(): row[f"PHabs_{m}"] = v
    row["PHabs_triangle_violation"] = triangle_violation_rate(Da, rng)
    row["PHsigned_triangle_violation"] = triangle_violation_rate(Ds, rng)
    return row, arr


def save_all(out):
    meta = pd.read_csv(CACHE / "meta.csv")
    out = sorted(out, key=lambda ra: ra[0]["idx"])
    df = meta.merge(pd.DataFrame([r for r, _ in out]), on="idx", validate="one_to_one")
    assert len(df) == len(meta), (len(df), len(meta))
    df.to_csv(CACHE / "features_scalar.csv", index=False)
    np.savez_compressed(CACHE / "features_arrays.npz", **{k: np.stack([a[k] for _, a in out]) for k in out[0][1]})
    print("saved", df.shape)


if __name__ == "__main__":
    # usage: 01_trial_features.py                 -> all trials, then save
    #        01_trial_features.py N               -> timing test on the first N trials
    #        01_trial_features.py --chunk I K     -> chunk I of K (parallel chunks), saved to cache/chunks/
    #        01_trial_features.py --merge K       -> merge K chunks and save
    import os, pickle
    mm_path = CACHE / "eeg_30s.npy"
    n = np.load(mm_path, mmap_mode="r").shape[0]
    n_jobs = int(os.environ.get("N_JOBS", 12))
    args = sys.argv[1:]
    if args[:1] == ["--merge"]:
        K = int(args[1]); out = []
        for i in range(K):
            with open(CACHE / "chunks" / f"chunk_{i:03d}.pkl", "rb") as fh: out += pickle.load(fh)
        save_all(out); sys.exit(0)
    if args[:1] == ["--chunk"]:
        I, K = int(args[1]), int(args[2])
        sel = np.array_split(np.arange(n), K)[I]
    else:
        sel = range(n) if not args else range(int(args[0]))
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5, batch_size=1)(delayed(one_trial)(int(i), mm_path) for i in sel)
    print(f"{len(out)} trials in {time.time()-t0:.1f}s")
    if args[:1] == ["--chunk"]:
        (CACHE / "chunks").mkdir(exist_ok=True)
        with open(CACHE / "chunks" / f"chunk_{I:03d}.pkl", "wb") as fh: pickle.dump(out, fh)
    elif not args:
        save_all(out)
