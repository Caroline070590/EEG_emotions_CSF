"""
synchrony_bands.py - band-resolved robustness companion to lib/synchrony.py.

Computes the SAME six metrics as lib.synchrony.synchrony_one() (PLV, PLI,
wPLI, amplitude-envelope correlation, max cross-correlation, lag), but
separately for:

  * the original broadband pass (1-45 Hz)      -> tag "broadband_1_45"
  * a delta-excluded broadband pass (4-45 Hz)  -> tag "broadband_4_45"
  * each canonical band (theta/alpha/beta/gamma) -> tag = band name

All bands other than delta clear the >=3-cycles-per-trial floor
established in diagnostic_band_reliability.py, so agreement between the
per-band effects and the original broadband effect indicates the
emotion-dependent synchronization results are not an artefact of
under-cycled low-frequency phase estimates.

Delta (1-4 Hz) is INTENTIONALLY still computed and reported (not silently
dropped) so that one can see directly how much the broadband estimate is
being pulled by the unreliable low end, rather than having it quietly
removed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, hilbert

from lib.config import CACHE
from lib.representations import load_raw_trial_csv, parse_subject_trial, _list_trials

FS = 250

BANDS = {
    "delta":           (1.0, 4.0),
    "theta":           (4.0, 8.0),
    "alpha":           (8.0, 13.0),
    "beta":            (13.0, 30.0),
    "gamma":           (30.0, 45.0),
    "broadband_1_45":  (1.0, 45.0),   # original pipeline band, for reference
    "broadband_4_45":  (4.0, 45.0),   # delta-excluded broadband, the recommended
                                       # drop-in replacement for the main analysis
}
SYNC_COLS = ["plv", "pli", "wpli", "amp_env_corr", "max_cross_corr", "lag_ms"]


def _bandpass(X, low, high, order=4, fs=FS):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=0)


def _triu(n):
    return np.triu_indices(n, 1)


def _metrics_from_filtered(Xf):
    """Same math as lib.synchrony.synchrony_one, factored out so it can be
    applied to any band-limited signal."""
    A = hilbert(Xf, axis=0)
    phase = np.angle(A); amp = np.abs(A)
    T, n = Xf.shape; iu = _triu(n)
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


def synchrony_one_multiband(path, bands=BANDS):
    X = load_raw_trial_csv(path)
    if X.shape[0] < 20:
        return None
    subj, tr = parse_subject_trial(path)
    out = dict(subject=subj, trial=Path(path).stem)
    for tag, (lo, hi) in bands.items():
        try:
            Xf = _bandpass(X, lo, hi)
            m = _metrics_from_filtered(Xf)
        except Exception:
            m = {c: np.nan for c in SYNC_COLS}
        for c in SYNC_COLS:
            out[f"{tag}__{c}"] = m[c]
    return out


def _one_with_emo(path, emo, bands):
    r = synchrony_one_multiband(path, bands)
    if r is not None:
        r["emotion"] = emo
    return r


def compute_synchrony_bands(bands=BANDS, n_jobs=-1, force=False):
    f = CACHE / "trial_synchrony_bands.csv"
    if f.exists() and not force:
        print("[synchrony_bands] cached"); return pd.read_csv(f)
    items = _list_trials()
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_one_with_emo)(p, emo, bands) for p, _, emo in items)
    df = pd.DataFrame([r for r in out if r is not None])
    df.to_csv(f, index=False)
    print(f"[synchrony_bands] {len(df)} trials -> {f}")
    return df
