"""
stats.py — repeated-measures statistics.

  * lmm_emotion_lrt        z(measure) ~ C(emotion) + (1|subject), ML LRT,
                           with OLS fallback if subject-variance collapses.
  * subject_cluster_bootstrap   resample SUBJECTS (correct unit), parallel.
  * pseudo_eta2 / valence_contrast / decoding helpers.
"""
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lib.config import EMOS, EI, VAL

# ----------------------------------------------------------------- LMM
def lmm_emotion_lrt(df, ycol):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from scipy.stats import chi2
    d = df[[ycol, "emotion", "subject"]].dropna().copy()
    d["emotion"] = pd.Categorical(d["emotion"], categories=EMOS)
    d["y"] = (d[ycol] - d[ycol].mean()) / d[ycol].std()
    dfd = len(EMOS) - 1
    for opt in ("lbfgs", "cg", "powell"):
        try:
            f = smf.mixedlm("y ~ C(emotion)", d, groups=d["subject"]).fit(reml=False, method=opt)
            n = smf.mixedlm("y ~ 1",          d, groups=d["subject"]).fit(reml=False, method=opt)
            lr = 2 * (f.llf - n.llf)
            return dict(stat=lr, df=dfd, p=float(chi2.sf(lr, dfd)), kind="LMM")
        except Exception:
            continue
    Xf = sm.add_constant(pd.get_dummies(d["emotion"], drop_first=True).astype(float))
    mf = sm.OLS(d["y"].values, Xf.values).fit()
    mn = sm.OLS(d["y"].values, np.ones((len(d), 1))).fit()
    lr = len(d) * np.log(mn.ssr / mf.ssr)
    return dict(stat=lr, df=dfd, p=float(chi2.sf(lr, dfd)), kind="OLS(reff~0)")

def lmm_table(tda, tcols):
    rows = []
    for c in tcols:
        r = lmm_emotion_lrt(tda, c); r["measure"] = c
        rows.append(r)
    return pd.DataFrame(rows)[["measure", "stat", "df", "p", "kind"]]

# ----------------------------------------------------------------- statistics
def pseudo_eta2(rows, Y, emo_idx, subj_codes, k):
    v = Y[rows, k].astype(float)
    dfc = pd.DataFrame({"v": v, "s": subj_codes[rows]})
    v = v - dfc.groupby("s")["v"].transform("mean").values
    em = np.array([v[emo_idx[rows] == j].mean() if (emo_idx[rows] == j).any() else 0
                   for j in range(len(EMOS))])
    return np.var(em, ddof=1) / (v.var() + 1e-12)

def valence_contrast(rows, agg, emo_idx):
    cent = np.vstack([agg[rows][emo_idx[rows] == EI[e]].mean(0) for e in EMOS])
    D = 1 - np.corrcoef(cent); np.fill_diagonal(D, 0)
    win = [D[i, j] for i, j in itertools.combinations(range(len(EMOS)), 2) if VAL[EMOS[i]] == VAL[EMOS[j]]]
    bet = [D[i, j] for i, j in itertools.combinations(range(len(EMOS)), 2) if VAL[EMOS[i]] != VAL[EMOS[j]]]
    return float(np.mean(bet) - np.mean(win))

# ----------------------------------------------------------------- bootstrap
def subject_cluster_bootstrap(am, statfun, n_boot=2000, n_jobs=-1, seed=42):
    """Resample SUBJECTS with replacement; recompute statfun(rows). Parallel."""
    subs = am["subject"].unique()
    sub_rows = {s: np.where(am["subject"].values == s)[0] for s in subs}
    obs = statfun(np.arange(len(am)))

    def one(b):
        rng = np.random.default_rng(seed + b)
        rows = np.concatenate([sub_rows[s] for s in rng.choice(subs, len(subs), replace=True)])
        return statfun(rows)

    boots = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(n_boot)))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(obs=obs, lo=lo, hi=hi, boots=boots)


def clustered_ols_emotion_contrasts(df, ycol, ref="neutral"):
    """Per-emotion contrasts using participant-standardized clustered OLS.

    This matches the manuscript's displayed emotion-versus-neutral contrast
    estimator. Multiple-testing correction is intentionally handled by the
    calling analysis because the correction family spans outcomes.
    """
    import statsmodels.formula.api as smf
    d = df[[ycol, "emotion", "subject"]].dropna().copy()
    means = d.groupby("subject")[ycol].transform("mean")
    sds = d.groupby("subject")[ycol].transform("std")
    # A participant whose measure has zero variance across videos (the lag of one participant) is assigned
    # standardized values of zero, so every contrast uses all participants (as in Supplementary Table S1).
    d["y"] = np.where(sds > 0, (d[ycol] - means) / sds.where(sds > 0, 1.0), 0.0)
    d["emotion"] = pd.Categorical(d["emotion"], categories=[ref] + [e for e in EMOS if e != ref])
    out = {ref: np.nan}
    m = smf.ols(f"y ~ C(emotion, Treatment('{ref}'))", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject"]}
    )
    pv = m.pvalues
    for e in EMOS:
        if e == ref:
            continue
        key = [k for k in pv.index if e in str(k)]
        out[e] = float(pv[key[0]]) if key else np.nan
    return out


# Compatibility alias retained so the historical plotting modules do not need
# a mechanical rename. The implementation is clustered OLS, not MixedLM.
lmm_emotion_contrasts = clustered_ols_emotion_contrasts


def emotion_slope_clustered_ols(df, ycol, xcol, ref="neutral"):
    """Fast emotion-specific y~x slopes for Figure 2 annotations.

    Inputs are already subject-z-scored in ``run_part2.py``.  We fit one compact
    interaction model with cluster-robust subject standard errors, then extract
    each emotion slope as a linear combination of the main x effect and the
    corresponding interaction coefficient.  This avoids repeated unstable
    MixedLM fits while retaining subject-cluster robust inference.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    d = df[[ycol, xcol, "emotion", "subject"]].dropna().copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    cats = [ref] + [e for e in EMOS if e != ref]
    d["emotion"] = pd.Categorical(d["emotion"], categories=cats)
    try:
        m = smf.ols(f"{ycol} ~ {xcol} * C(emotion)", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["subject"]})
    except Exception:
        return {e: (np.nan, np.nan) for e in EMOS}, np.nan

    names = list(m.params.index)
    b = m.params.values
    cov = m.cov_params().loc[names, names].values
    def idx(nm): return names.index(nm) if nm in names else None
    slopes = {}
    ix = idx(xcol)
    for e in EMOS:
        if ix is None:
            slopes[e] = (np.nan, np.nan)
            continue
        c = np.zeros(len(names)); c[ix] = 1.0
        if e != ref:
            poss = [f"{xcol}:C(emotion)[T.{e}]", f"C(emotion)[T.{e}]:{xcol}"]
            it = next((idx(nm) for nm in poss if idx(nm) is not None), None)
            if it is not None:
                c[it] = 1.0
        est = float(c @ b)
        se = float(np.sqrt(max(c @ cov @ c, 1e-18)))
        p = float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan
        slopes[e] = (est, p)

    inter_terms = [i for i,nm in enumerate(names) if (f":{xcol}" in nm or f"{xcol}:" in nm) and "C(emotion)" in nm]
    if inter_terms:
        R = np.zeros((len(inter_terms), len(names)))
        for r, i in enumerate(inter_terms):
            R[r, i] = 1.0
        try:
            interaction_p = float(m.wald_test(R, scalar=True).pvalue)
        except Exception:
            interaction_p = np.nan
    else:
        interaction_p = np.nan
    return slopes, interaction_p


# Historical name retained for importing modules; this estimator is OLS.
emotion_slope_lmm = emotion_slope_clustered_ols


def boot_vs_neutral(df, col, emo, ref="neutral", n_boot=2000, seed=0):
    """Paired subject-cluster bootstrap of the standardized mean difference
    (emotion - ref) for `col`. Resamples SUBJECTS. Returns (d_obs, lo, hi)."""
    piv = df.groupby(["subject", "emotion"])[col].mean().unstack()
    if emo not in piv or ref not in piv: return (np.nan, np.nan, np.nan)
    d = (piv[emo] - piv[ref]).dropna()
    sd = pd.concat([piv[emo], piv[ref]]).std()
    if sd == 0 or len(d) < 5: return (np.nan, np.nan, np.nan)
    obs = d.mean() / sd
    rng = np.random.default_rng(seed); vals = d.values; n = len(vals)
    boots = np.array([np.mean(rng.choice(vals, n, replace=True)) for _ in range(n_boot)]) / sd
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(obs), float(lo), float(hi)
