"""
run_part2.py — Figure 2: EEG synchronization and synchrony-topology coupling.
a  synchrony fingerprint (emotion x metric, column z; per-cell vs-neutral stars)
b  pooled synchrony-topology coupling (distinct colormap)
c  coupling strength per emotion (mean |beta| over all topology x synchrony pairs)
d-g emotion-specific coupling per topology metric (interaction-LMM per-emotion slopes)
No baked-in figure title (goes in the manuscript caption).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
from lib.config import (CACHE, FIG_DIR, EMOS, EMO_COLORS, set_nature_style, save, W2,
                        panel_letter, nice_heatmap, slim_colorbar)
from lib.stats import lmm_emotion_contrasts, emotion_slope_lmm
from lib.synchrony import SYNC_COLS

SLAB = {"plv":"PLV","pli":"PLI","wpli":"wPLI","amp_env_corr":"AEC",
        "max_cross_corr":"MaxXC","lag_ms":"Lag"}
TOPO = ["sigma_small_world","transitivity","char_path_len_gcc","global_efficiency"]
TLAB = {"sigma_small_world":"σ","transitivity":"Transitivity",
        "char_path_len_gcc":"Path length","global_efficiency":"Efficiency"}
def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<.05 else ""
def _zw(df, cols):
    o=df.copy()
    for c in cols: o[c+"_z"]=o.groupby("subject")[c].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
    return o
def _pooled_beta(d,t,s):
    dd=d[[t,s,"subject"]].dropna()
    if dd[s].std()<1e-9: return np.nan,np.nan
    try:
        m=smf.ols(f"{t} ~ {s}",dd).fit(cov_type="cluster", cov_kwds={"groups": dd["subject"]})
        return float(m.params[s]),float(m.pvalues[s])
    except Exception: return np.nan,np.nan

def main():
    set_nature_style()
    fs=CACHE/"trial_synchrony_metrics.csv"; ft=CACHE/"trial_metrics_spearman.csv"
    if not (fs.exists() and ft.exists()): print("  [skip] need synchrony + spearman CSVs"); return
    sync=pd.read_csv(fs); topo=pd.read_csv(ft)
    # Merge at the trial level.  Merging only by subject+emotion creates a
    # Cartesian product across repeated trials and inflates the coupling table.
    df = pd.merge(sync, topo, on=["subject", "emotion", "trial"],
                  how="inner", validate="one_to_one")
    df = _zw(df, TOPO + SYNC_COLS)
    df.to_csv(CACHE / "trial_sync_topology_merged_spearman.csv", index=False)

    # ---- precompute emotion-specific coupling (slopes + significance) ----
    Be_all={}; Pe_all={}; inter_all={}
    for t in TOPO:
        Be=np.full((len(EMOS),len(SYNC_COLS)),np.nan); Pe=np.full_like(Be,np.nan); inter=[""]*len(SYNC_COLS)
        for j,s in enumerate(SYNC_COLS):
            slopes,pir=emotion_slope_lmm(df,t+"_z",s+"_z",ref="neutral")
            for ei,e in enumerate(EMOS): Be[ei,j],Pe[ei,j]=slopes[e]
            if np.isfinite(pir) and pir<0.05: inter[j]="†"
        Be_all[t]=Be; Pe_all[t]=Pe; inter_all[t]=inter
    # coupling strength per emotion = mean |beta| over all topo x sync; sig count
    strength=np.zeros(len(EMOS)); sigcount=np.zeros(len(EMOS),int)
    for t in TOPO:
        strength += np.nanmean(np.abs(Be_all[t]),axis=1)
        sigcount += (Pe_all[t]<0.05).sum(axis=1)
    strength/=len(TOPO)

    fig=plt.figure(figsize=(W2, W2*0.86))
    gs=GridSpec(3,4,figure=fig,hspace=0.95,wspace=0.62,top=0.95,bottom=0.08,
                height_ratios=[1.0,0.72,1.0])

    # a: fingerprint
    axA=fig.add_subplot(gs[0,0:3])
    sub=sync.groupby(["subject","emotion"])[SYNC_COLS].mean().reset_index()
    heat=sub.groupby("emotion")[SYNC_COLS].mean().reindex(EMOS); z=(heat-heat.mean(0))/heat.std(0)
    im=nice_heatmap(axA,z.values,EMOS,[SLAB[c] for c in SYNC_COLS],cmap="magma",
                    vmin=-2,vmax=2,aspect="auto",fontsize=6.0)
    for j,c in enumerate(SYNC_COLS):
        pmap=lmm_emotion_contrasts(sync,c,ref="neutral")
        for i,e in enumerate(EMOS):
            st=stars(pmap.get(e,np.nan)) if e!="neutral" else ""
            if st: axA.text(j,i,st,ha="center",va="center",fontsize=5,
                            color="white" if z.values[i,j]<0.5 else "#111111")
    axA.set_title("Synchrony fingerprint  (cells: emotion vs neutral)", pad=6)
    panel_letter(axA,"a",dx=-0.06); slim_colorbar(fig,im,axA,"column z")

    # b: pooled coupling — DISTINCT colormap (PuOr_r)
    axB=fig.add_subplot(gs[1,0:2])
    B=np.full((len(TOPO),len(SYNC_COLS)),np.nan); P=np.full_like(B,np.nan)
    for i,t in enumerate(TOPO):
        for j,s in enumerate(SYNC_COLS): B[i,j],P[i,j]=_pooled_beta(df,t+"_z",s+"_z")
    im2=nice_heatmap(axB,B,[TLAB[t] for t in TOPO],[SLAB[c] for c in SYNC_COLS],
                     cmap="PuOr_r",vmin=-0.3,vmax=0.3,aspect="auto",fontsize=5.8)
    for i in range(len(TOPO)):
        for j in range(len(SYNC_COLS)):
            if stars(P[i,j]): axB.text(j,i,stars(P[i,j]),ha="center",va="center",fontsize=5,
                                       color="white" if abs(B[i,j])>0.18 else "#111111")
    axB.set_title("Pooled coupling (β)", pad=6); panel_letter(axB,"b",dx=-0.12)
    slim_colorbar(fig,im2,axB,"β")

    # c: coupling strength per emotion (summary bar)
    axC=fig.add_subplot(gs[1,2:4])
    order=np.argsort(strength)[::-1]
    axC.barh(np.arange(len(EMOS)), strength[order],
             color=[EMO_COLORS[EMOS[i]] for i in order], edgecolor="k", linewidth=0.3)
    for k,i in enumerate(order):
        if sigcount[i]>0: axC.text(strength[i], k, f"  {sigcount[i]} sig", va="center", fontsize=4.8, color="0.3")
    axC.set_yticks(range(len(EMOS))); axC.set_yticklabels([EMOS[i] for i in order], fontsize=5.4)
    axC.invert_yaxis(); axC.set_xlabel("mean |β| coupling strength")
    axC.set_title("Coupling strength per emotion", pad=6)
    for sp in ["top","right"]: axC.spines[sp].set_visible(False)
    panel_letter(axC,"c",dx=-0.14)

    # d-g: emotion-specific coupling
    for k,t in enumerate(TOPO):
        ax=fig.add_subplot(gs[2,k]); Be=Be_all[t]; Pe=Pe_all[t]
        col_labels=[SLAB[c]+("\n"+inter_all[t][j] if inter_all[t][j] else "") for j,c in enumerate(SYNC_COLS)]
        im3=nice_heatmap(ax,Be,EMOS,col_labels,cmap="RdBu_r",vmin=-0.2,vmax=0.2,aspect="auto",fontsize=5.0)
        for ei in range(len(EMOS)):
            for j in range(len(SYNC_COLS)):
                if stars(Pe[ei,j]): ax.text(j,ei,stars(Pe[ei,j]),ha="center",va="center",
                                            fontsize=4.0,color="white" if abs(Be[ei,j])>0.12 else "#111111")
        ax.set_title(TLAB[t], pad=4); panel_letter(ax,"defg"[k])
        if k==3: slim_colorbar(fig,im3,ax,"β per emotion")

    save(fig, FIG_DIR/"part2"/"Figure_Part2_synchrony_coupling"); plt.close(fig)

if __name__ == "__main__":
    main()
