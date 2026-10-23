"""Recurrence quantification (RQA) and joint-recurrence synchronization on the 30-s trials.

Fixed, pre-specified parameters (not tuned):
  * signal decimated 250 -> 125 Hz (zero-phase anti-alias filter), z-scored per window
  * ten non-overlapping 3-s windows per trial (375 samples); results averaged over windows
  * time-delay embedding m = 3, tau = 3 samples (24 ms); Euclidean distances
  * threshold set per channel and window to give a fixed recurrence rate RR = 5%
    (Theiler window of 2 samples excluded), so RQA measures are comparable across trials
  * l_min = v_min = 2
Per channel: DET (determinism), LAM (laminarity), L_mean (mean diagonal line), ENTR (Shannon
entropy of diagonal line lengths), TT (trapping time).
Per channel pair: joint-recurrence synchronization S_cd = RR(R_c AND R_d) / RR (Romano et al.,
2004): 0.05 for independent recurrences, 1 for identical ones; captures generalized (nonlinear)
synchronization.

usage: 02_recurrence_features.py --chunk I K | --merge K
"""
import sys, os, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.signal import decimate
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parents[1]; CACHE = HERE / "cache"
M, TAU, RR, THEILER, WIN, NWIN = 3, 3, 0.05, 2, 375, 10
IU = np.triu_indices(30, 1)
RQA = ["DET", "LAM", "Lmean", "ENTR", "TT"]

def embed(x):
    n = len(x) - (M - 1) * TAU
    return np.stack([x[i * TAU: i * TAU + n] for i in range(M)], 1)

def recurrence(x):
    D = squareform(pdist(embed(x)))
    n = len(D); mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) > THEILER
    eps = np.quantile(D[mask], RR)
    return (D <= eps) & mask

def _lines(R):
    """Lengths of diagonal (upper triangle, symmetric) and vertical lines."""
    n = len(R); diag, vert = [], []
    for k in range(THEILER + 1, n):
        d = np.diagonal(R, k).astype(np.int8)
        if d.any():
            e = np.diff(np.r_[0, d, 0]); diag += list(np.where(e == -1)[0] - np.where(e == 1)[0])
    for j in range(n):
        c = R[:, j].astype(np.int8)
        if c.any():
            e = np.diff(np.r_[0, c, 0]); vert += list(np.where(e == -1)[0] - np.where(e == 1)[0])
    return np.array(diag), np.array(vert)

def rqa(R):
    dl, vl = _lines(R)
    n_rec_upper = R[np.triu_indices(len(R), THEILER + 1)].sum()
    dl2, vl2 = dl[dl >= 2], vl[vl >= 2]
    det = dl2.sum() / n_rec_upper if n_rec_upper else np.nan
    lam = vl2.sum() / R.sum() if R.sum() else np.nan
    if len(dl2):
        _, cnt = np.unique(dl2, return_counts=True); p = cnt / cnt.sum(); ent = float(-(p * np.log(p)).sum())
    else:
        ent = 0.0
    return [det, lam, dl2.mean() if len(dl2) else 0.0, ent, vl2.mean() if len(vl2) else 0.0]

def one_trial(idx):
    X = np.load(CACHE / "eeg_30s.npy", mmap_mode="r")[idx].astype(np.float64)
    Xd = decimate(X, 2, axis=1, zero_phase=True)          # 30 x 3750
    feats = np.zeros((NWIN, 30, len(RQA))); jr = np.zeros((NWIN, len(IU[0])))
    for w in range(NWIN):
        seg = Xd[:, w * WIN:(w + 1) * WIN]
        seg = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-12)
        Rs = [recurrence(seg[c]) for c in range(30)]
        for c, R in enumerate(Rs): feats[w, c] = rqa(R)
        F = np.stack([R.ravel() for R in Rs]).astype(np.float32)
        J = (F @ F.T) / F.shape[1]                         # joint recurrence rate
        jr[w] = (J / RR)[IU]
    return idx, np.nanmean(feats, 0).astype(np.float32), jr.mean(0).astype(np.float32)

if __name__ == "__main__":
    n = np.load(CACHE / "eeg_30s.npy", mmap_mode="r").shape[0]
    a = sys.argv[1:]
    if a[0] == "--chunk":
        I, K = int(a[1]), int(a[2]); sel = np.array_split(np.arange(n), K)[I]
        t0 = time.time()
        out = Parallel(n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), batch_size=1)(delayed(one_trial)(int(i)) for i in sel)
        (CACHE / "rqa_chunks").mkdir(exist_ok=True)
        pickle.dump(out, open(CACHE / "rqa_chunks" / f"chunk_{I:03d}.pkl", "wb"))
        print(f"{len(out)} trials in {time.time() - t0:.0f}s")
    elif a[0] == "--merge":
        out = []
        for i in range(int(a[1])): out += pickle.load(open(CACHE / "rqa_chunks" / f"chunk_{i:03d}.pkl", "rb"))
        out.sort(key=lambda t: t[0]); assert [t[0] for t in out] == list(range(n))
        np.savez_compressed(CACHE / "recurrence_features.npz",
                            rqa=np.stack([t[1] for t in out]), jrs=np.stack([t[2] for t in out]), rqa_names=np.array(RQA))
        print("saved", n)
    elif a[0] == "--test":
        t0 = time.time(); i, f, j = one_trial(0)
        print(f"one trial {time.time() - t0:.1f}s | RQA ch0 {dict(zip(RQA, f[0].round(3)))} | JRS mean {j.mean():.3f} min {j.min():.3f} max {j.max():.3f}")
