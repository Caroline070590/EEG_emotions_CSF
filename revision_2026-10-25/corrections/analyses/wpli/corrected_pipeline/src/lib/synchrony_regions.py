"""
synchrony_regions.py — regional synchronization metrics, companion to
lib/synchrony.py (which computes PLV/PLI/wPLI/AEC/MaxXC/Lag pooled over
all 30 channels). Here the SAME six metrics are computed restricted to
the channel PAIRS WITHIN each anatomical region (frontal/central/
temporal/parietal/occipital), using the exact same band-pass + Hilbert
pipeline as the main Part 2 synchrony analysis.

Note: temporal has only 2 channels (T7, T8) -> exactly 1 pair per trial.
That single pair is still a valid (if less statistically powerful)
region-restricted synchrony estimate; it is not dropped, but its lower
reliability should be kept in mind when interpreting results for that
region specifically.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, hilbert

from lib.config import CH, CACHE
from lib.representations import load_raw_trial_csv, parse_subject_trial, _list_trials
from lib.regions import REGIONS, REGION_MAP

FS = 250
SYNC_COLS = ["plv", "pli", "wpli", "amp_env_corr", "max_cross_corr", "lag_ms"]
COLIDX = {c: i for i, c in enumerate(CH)}
REGION_IDX = {r: [COLIDX[c] for c in chans if c in COLIDX] for r, chans in REGION_MAP.items()}


def _bandpass(X, low=1.0, high=45.0, order=4, fs=FS):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=0)


def _metrics_from_signal(Xf):
    """Same math as lib.synchrony.synchrony_one, applied to an arbitrary
    channel subset Xf (T x n_region_channels)."""
    A = hilbert(Xf, axis=0)
    phase = np.angle(A); amp = np.abs(A)
    T, n = Xf.shape
    if n < 2:
        return {c: np.nan for c in SYNC_COLS}
    iu = np.triu_indices(n, 1)
    z = np.exp(1j * phase)
    S = (z.conj().T @ z) / T
    plv = np.abs(S)[iu]
    CS = A[:, :, None] * np.conj(A[:, None, :])
    Im = np.imag(CS)
    pli = np.abs(np.mean(np.sign(Im), axis=0))[iu]
    denom = np.mean(np.abs(Im), axis=0)
    num = np.abs(np.mean(Im, axis=0))
    wpli = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)[iu]
    aenv = amp - amp.mean(0); aenv /= (aenv.std(0) + 1e-12)
    aec = (aenv.T @ aenv) / T; amp_env = aec[iu]
    Xn = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12)
    F = np.fft.rfft(Xn, n=2 * T, axis=0)
    mx = np.empty(len(iu[0])); lag = np.empty(len(iu[0]))
    for idx, (i, j) in enumerate(zip(*iu)):
        cc = np.fft.irfft(F[:, i] * np.conj(F[:, j]), n=2 * T) / T
        cc = np.concatenate([cc[-(T - 1):], cc[:T]])
        k = np.argmax(np.abs(cc)); mx[idx] = cc[k]; lag[idx] = (k - (T - 1)) / FS * 1000
    return dict(plv=plv.mean(), pli=pli.mean(), wpli=wpli.mean(),
                amp_env_corr=amp_env.mean(), max_cross_corr=np.abs(mx).mean(),
                lag_ms=np.abs(lag).mean())


def synchrony_regions_one(path, emo):
    X = load_raw_trial_csv(path)
    if X.shape[0] < 20:
        return []
    subj, tr = parse_subject_trial(path)
    Xf_full = _bandpass(X)   # band-pass once, then subset per region (cheaper)
    rows = []
    for region, idx in REGION_IDX.items():
        m = _metrics_from_signal(Xf_full[:, idx])
        rows.append(dict(subject=subj, emotion=emo, trial=Path(path).stem,
                          region=region, n_channels=len(idx), **m))
    return rows


def compute_synchrony_regions(n_jobs=1, force=False):
    f = CACHE / "region_trial_synchrony.csv"
    if f.exists() and not force:
        print("[synchrony_regions] cached"); return pd.read_csv(f)
    items = _list_trials(); t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(synchrony_regions_one)(p, emo) for p, _, emo in items)
    rows = [r for sub in out for r in sub]
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False)
    print(f"[synchrony_regions] {len(items)} trials x {len(REGIONS)} regions in {time.time()-t0:.1f}s")
    return df
