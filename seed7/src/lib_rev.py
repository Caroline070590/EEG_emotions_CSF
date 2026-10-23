"""Shared statistics and plotting helpers for the revision analyses.

Primary inference treats BOTH participants and video stimuli as random factors.
SEED-VII version (identical tests; only the design constants differ). SEED-VII is fully balanced
(20 participants x 80 clips, clips nested in condition), so the same min F' (Clark, 1973) applies:
  F1 (by participant): paired test on participant x condition means, df1 = 19
  F2 (by clip):        clip means (averaged over participants); error = between-clip
                       variance within condition, pooled over the 7 conditions, df2 = 80 - 7 = 73
The `video` column is the clip index (0-79); `source` groups clips cut from the same programme.
  min F' = F1 F2 / (F1 + F2), with denominator df (F1+F2)^2 / (F1^2/df2 + F2^2/df1).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
CACHE, FIG, TAB = HERE / "cache", HERE / "figures", HERE / "tables"
EMOS = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'happiness', 'surprise']
NONNEU = [e for e in EMOS if e != 'neutral']
EMO_COLORS = {'anger': '#c0392b', 'disgust': '#7d3c98', 'fear': '#34495e', 'sadness': '#2471a3',
              'neutral': '#7f8c8d', 'happiness': '#f1c40f', 'surprise': '#16a085'}
CH = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ', 'C3', 'C4',
      'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2']
TOPO = ["transitivity", "global_efficiency", "char_path_len_gcc",
        "gamma_C_over_Crand", "lambda_L_over_Lrand", "sigma_small_world"]
TOPO_LABEL = {"transitivity": "Transitivity", "global_efficiency": "Efficiency",
              "char_path_len_gcc": "Path length", "gamma_C_over_Crand": "γ",
              "lambda_L_over_Lrand": "λ", "sigma_small_world": "σ"}
BANDS5 = ["delta", "theta", "alpha", "beta", "gamma"]
W1, W15, W2 = 3.50, 4.72, 7.20


def load():
    df = pd.read_csv(CACHE / "features_scalar.csv")
    arr = dict(np.load(CACHE / "features_arrays.npz"))
    assert len(df) == 1600 and df.groupby(["subject", "video"]).size().max() == 1
    return df, arr


# ------------------------------------------------------------------ min F'
def _cell_means(y, df, unit="video"):
    """unit: "video" (clip, primary) or "source" (programme; sources spanning two conditions are split)."""
    d = pd.DataFrame({"y": y, "subject": df.subject.values, "video": df[unit].values, "emotion": df.emotion.values})
    subj = d.groupby(["subject", "emotion"]).y.mean().unstack()[EMOS]       # 20 x 7
    vid = d.groupby(["video", "emotion"]).y.mean().reset_index()            # 80 rows
    return subj, vid

def _minf(F1, df1, F2, df2):
    if not (np.isfinite(F1) and np.isfinite(F2)) or F1 + F2 == 0:
        return np.nan, np.nan
    mf = F1 * F2 / (F1 + F2)
    dfd = (F1 + F2) ** 2 / (F1 ** 2 / df2 + F2 ** 2 / df1)
    return mf, dfd

def contrast_vs_neutral(y, df, emo, unit="video"):
    """Emotion-vs-neutral with participants and stimuli random. Returns a dict."""
    subj, vid = _cell_means(np.asarray(y, float), df, unit)
    diff = (subj[emo] - subj["neutral"]).dropna()
    n = len(diff); dz = diff.mean() / diff.std(ddof=1)
    t1 = diff.mean() / (diff.std(ddof=1) / np.sqrt(n)); df1 = n - 1
    vid = vid.copy(); vid["dev"] = vid.y - vid.groupby("emotion").y.transform("mean")
    df2 = len(vid) - vid.emotion.nunique()
    mse = (vid.dev ** 2).sum() / df2
    a, b = vid.loc[vid.emotion == emo, "y"], vid.loc[vid.emotion == "neutral", "y"]
    t2 = (a.mean() - b.mean()) / np.sqrt(mse * (1 / len(a) + 1 / len(b)))
    mf, dfd = _minf(t1 ** 2, df1, t2 ** 2, df2)
    return {"emotion": emo, "diff": diff.mean(), "dz": dz,
            "t1": t1, "p1": 2 * stats.t.sf(abs(t1), df1),
            "t2": t2, "p2": 2 * stats.t.sf(abs(t2), df2),
            "minF": mf, "minF_df": dfd, "p_minF": stats.f.sf(mf, 1, dfd) if np.isfinite(mf) else np.nan}

def omnibus(y, df):
    """Omnibus emotion effect: RM-ANOVA by participant (F1), one-way ANOVA on video means (F2), min F'."""
    subj, vid = _cell_means(np.asarray(y, float), df)
    S = subj.dropna().to_numpy(); n, k = S.shape
    gm = S.mean(); ss_e = n * ((S.mean(0) - gm) ** 2).sum()
    ss_s = k * ((S.mean(1) - gm) ** 2).sum(); ss_t = ((S - gm) ** 2).sum()
    ss_err = ss_t - ss_e - ss_s; d1n, d1d = k - 1, (k - 1) * (n - 1)
    F1 = (ss_e / d1n) / (ss_err / d1d)
    groups = [g.y.values for _, g in vid.groupby("emotion")]
    F2, _ = stats.f_oneway(*groups); d2d = len(vid) - len(groups)
    mf = F1 * F2 / (F1 + F2)
    dfd = (F1 + F2) ** 2 / (F1 ** 2 / d2d + F2 ** 2 / d1d)
    return {"F1": F1, "p1": stats.f.sf(F1, d1n, d1d), "F2": F2, "p2": stats.f.sf(F2, d1n, d2d),
            "minF": mf, "minF_df": dfd, "p_minF": stats.f.sf(mf, d1n, dfd)}

def bh(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum():
        from statsmodels.stats.multitest import multipletests
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


# ------------------------------------------------------------------ plotting
def style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "figure.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "axes.titlepad": 4,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "axes.edgecolor": "#222222",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False})

def panel(ax, letter):
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("saved", FIG / f"{name}.pdf")

DIVERGING = "RdBu_r"   # two hues, neutral light midpoint


# ------------------------------------------------------------------ source-level sensitivity (SEED-VII)
def contrast_vs_neutral_source_nested(y, df, emo):
    """Emotion-vs-neutral min F' with source programmes as the stimulus unit, nested within condition.
    Clips from the programmes that contributed to two conditions are excluded, so every remaining programme
    belongs to one condition. Both components weight programmes equally: F1 uses, per participant, the mean
    over programmes of the participant's mean over that programme's clips; F2 uses programme means (over
    participants and clips) with between-programme variance pooled within condition (df2 = n_programmes - 7)."""
    d = pd.DataFrame({"y": np.asarray(y, float), "subject": df.subject.values, "source": df.source.values, "emotion": df.emotion.values})
    mixed = d.groupby("source").emotion.nunique(); d = d[~d.source.isin(mixed[mixed > 1].index)]
    ps = d.groupby(["subject", "emotion", "source"]).y.mean().reset_index()
    subj = ps.groupby(["subject", "emotion"]).y.mean().unstack()
    diff = (subj[emo] - subj["neutral"]).dropna(); n = len(diff)
    t1 = diff.mean() / (diff.std(ddof=1) / np.sqrt(n)); df1 = n - 1
    src = d.groupby(["source", "emotion"]).y.mean().reset_index()
    src["dev"] = src.y - src.groupby("emotion").y.transform("mean")
    df2 = len(src) - src.emotion.nunique(); mse = (src.dev ** 2).sum() / df2
    a, b = src.loc[src.emotion == emo, "y"], src.loc[src.emotion == "neutral", "y"]
    t2 = (a.mean() - b.mean()) / np.sqrt(mse * (1 / len(a) + 1 / len(b)))
    mf, dfd = _minf(t1 ** 2, df1, t2 ** 2, df2)
    return {"emotion": emo, "dz_source_weighted": diff.mean() / diff.std(ddof=1), "t1": t1, "t2": t2, "df2": df2,
            "p_minF": stats.f.sf(mf, 1, dfd) if np.isfinite(mf) else np.nan, "n_sources": len(src),
            "n_sources_emo": len(a), "n_sources_neutral": len(b)}
