"""
synchrony.py — Part 2 EEG synchronization metrics (vectorised, parallelised).

Per trial, on a 1-45 Hz band-pass + Hilbert analytic signal, compute the mean
(over channel pairs) of:
  PLV, PLI                  (phase-difference descriptors)
  wPLI                      (amplitude-weighted analytic cross-spectrum)
  amp_env_corr             (amplitude-envelope correlation)
  max_cross_corr, lag_ms   (temporal coupling)
Coherence / mutual information are available but OFF by default (slow); enable
with heavy=True.
"""
from pathlib import Path
import time
import hashlib
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, hilbert

from lib.config import CH, EMOS, VAL, CACHE
from lib.representations import load_raw_trial_csv, parse_subject_trial, _list_trials

FS = 250
ESTIMATOR_VERSION = "analytic_cross_spectrum_wpli_v1"
SYNC_COLS = ["plv","pli","wpli","amp_env_corr","max_cross_corr","lag_ms"]

def _bandpass(X, low=1.0, high=45.0, order=4, fs=FS):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, X, axis=0)

def _triu(n): return np.triu_indices(n, 1)

def synchrony_one(path):
    X = load_raw_trial_csv(path)
    if X.shape[0] < 20: return None
    subj, tr = parse_subject_trial(path)
    Xf = _bandpass(X)
    A = hilbert(Xf, axis=0)                 # analytic signal  (T x n)
    phase = np.angle(A); amp = np.abs(A)
    T, n = Xf.shape; iu = _triu(n)
    z = np.exp(1j * phase)                   # T x n
    # PLV = |<exp(i dphi)>|
    S = (z.conj().T @ z) / T                 # n x n complex
    plv = np.abs(S)[iu]
    # PLI / wPLI from imaginary part of cross-spectrum per sample
    # build pairwise imaginary mean via einsum (T x n x n is big; loop pairs in chunks)
    Im = np.imag(A[:, :, None] * np.conj(A)[:, None, :])   # T x n x n; amplitude weights retained
    pli = np.abs(np.mean(np.sign(Im), axis=0))[iu]
    denom = np.mean(np.abs(Im), axis=0); num = np.abs(np.mean(Im, axis=0))
    wpli = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)[iu]
    # amplitude-envelope correlation
    aenv = amp - amp.mean(0); aenv /= (aenv.std(0) + 1e-12)
    aec = (aenv.T @ aenv) / T; amp_env = aec[iu]
    # max cross-correlation + lag (FFT-based), per pair
    Xn = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12)
    F = np.fft.rfft(Xn, n=2*T, axis=0)
    mx = np.empty(len(iu[0])); lag = np.empty(len(iu[0]))
    for idx, (i, j) in enumerate(zip(*iu)):
        cc = np.fft.irfft(F[:, i] * np.conj(F[:, j]), n=2*T) / T
        cc = np.concatenate([cc[-(T-1):], cc[:T]])
        k = np.argmax(np.abs(cc)); mx[idx] = cc[k]; lag[idx] = (k - (T-1)) / FS * 1000
    return dict(subject=subj, emotion=None, trial=Path(path).stem,
                plv=plv.mean(), pli=pli.mean(), wpli=wpli.mean(),
                amp_env_corr=amp_env.mean(), max_cross_corr=np.abs(mx).mean(),
                lag_ms=np.abs(lag).mean())

def _one_with_emo(path, emo):
    r = synchrony_one(path)
    if r is not None: r["emotion"] = emo
    return r

def compute_synchrony(n_jobs=-1, force=False):
    f = CACHE / "trial_synchrony_metrics.csv"
    metadata = CACHE / "trial_synchrony_metrics.metadata.json"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if f.exists() and not force:
        try:
            info = json.loads(metadata.read_text())
            valid = (info.get("estimator_version") == ESTIMATOR_VERSION
                     and info.get("source_sha256") == source_sha
                     and info.get("csv_sha256") == hashlib.sha256(f.read_bytes()).hexdigest())
        except (OSError, ValueError):
            valid = False
        if valid:
            print("[synchrony] verified amplitude-weighted cache"); return pd.read_csv(f)
        print("[synchrony] recomputing cache without matching estimator provenance")
    items = _list_trials(); t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one_with_emo)(p, emo) for p, _, emo in items)
    df = pd.DataFrame([r for r in out if r is not None])
    df.to_csv(f, index=False)
    metadata.write_text(json.dumps({"estimator_version": ESTIMATOR_VERSION,
                                   "source_sha256": source_sha,
                                   "csv_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                   "n_trials": len(df)}, indent=2) + "\n")
    print(f"[synchrony] {len(df)} trials in {time.time()-t0:.1f}s")
    return df
