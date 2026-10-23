"""
run_sample_size.py — sample-size / data-quantity robustness.
a  decoding balanced accuracy vs number of SUBJECTS (learning curve; is it saturated?).
b  RSA split-half reliability vs number of subjects (geometry stabilization).
c  representation convergence vs number of TIME SAMPLES per trial: similarity of the
   truncated |Spearman| matrix to the full 250-sample matrix (does 250 suffice?).
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from lib.config import CACHE, FIG_DIR, EMOS, CH, set_nature_style, save, W2, panel_letter
from lib.representations import _list_trials, load_raw_trial_csv

def decode_acc(X, y, groups, folds=5):
    gkf=GroupKFold(min(folds, len(np.unique(groups))))
    yt,yp=[],[]
    for tr,te in gkf.split(X,y,groups):
        sc=StandardScaler().fit(X[tr])
        clf=LogisticRegression(max_iter=1500).fit(sc.transform(X[tr]),y[tr])
        yp.append(clf.predict(sc.transform(X[te]))); yt.append(y[te])
    return balanced_accuracy_score(np.concatenate(yt),np.concatenate(yp))

def rdm(agg, am, subs):
    C=np.vstack([agg[((am.emotion==e).values)&(am.subject.isin(subs).values)].mean(0) for e in EMOS])
    return squareform(pdist(C, metric="correlation"))

def main():
    set_nature_style()
    agg=np.load(CACHE/"agg_spearman.npy"); am=pd.read_csv(CACHE/"agg_meta_spearman.csv")
    subs=np.array(sorted(am.subject.unique())); rng=np.random.default_rng(0)
    iu=np.triu_indices(len(EMOS),1)

    # (a) learning curve vs subjects
    ks=[20,40,60,80,100,len(subs)]; A_m=[]; A_lo=[]; A_hi=[]
    for k in ks:
        accs=[]
        for _ in range(8 if k<len(subs) else 1):
            sel=rng.choice(subs,k,replace=False) if k<len(subs) else subs
            m=am.subject.isin(sel).values
            accs.append(decode_acc(agg[m],am.emotion.values[m],am.subject.values[m]))
        A_m.append(np.mean(accs)); A_lo.append(np.percentile(accs,5) if len(accs)>1 else accs[0]); A_hi.append(np.percentile(accs,95) if len(accs)>1 else accs[0])

    # (b) RSA reliability vs subjects (split-half within k subjects)
    B_m=[]
    for k in ks:
        rs=[]
        for _ in range(40):
            sel=rng.choice(subs,k,replace=False) if k<len(subs) else subs.copy()
            rng.shuffle(sel); h=len(sel)//2
            R1=rdm(agg,am,sel[:h]); R2=rdm(agg,am,sel[h:])
            rs.append(spearmanr(R1[iu],R2[iu]).correlation)
        B_m.append(np.nanmean(rs))

    # (c) representation convergence vs time samples (subset of trials)
    trials=_list_trials(); idx=rng.choice(len(trials),200,replace=False)
    Ts=[50,100,150,200,250]; conv={t:[] for t in Ts}
    iuC=np.triu_indices(len(CH),1)
    for ii in idx:
        path=trials[ii][0]
        try: full=load_raw_trial_csv(path)
        except Exception: continue
        if full.shape[0] < 250: continue
        base=pd.DataFrame(full[:250]).corr(method="spearman").abs().values.copy(); np.fill_diagonal(base,0)
        for t in Ts:
            sub=pd.DataFrame(full[:t]).corr(method="spearman").abs().values.copy(); np.fill_diagonal(sub,0)
            conv[t].append(spearmanr(sub[iuC], base[iuC]).correlation)
    C_m=[np.nanmean(conv[t]) for t in Ts]; C_sd=[np.nanstd(conv[t]) for t in Ts]

    fig=plt.figure(figsize=(W2, W2*0.32)); gs=GridSpec(1,3,figure=fig,wspace=0.45,top=0.84,bottom=0.22)
    axA=fig.add_subplot(gs[0,0])
    axA.fill_between(ks,A_lo,A_hi,color="#4c72b0",alpha=0.2)
    axA.plot(ks,A_m,"o-",color="#4c72b0",ms=4,lw=1.3)
    axA.axhline(1/9,color="0.5",ls="--",lw=0.7); axA.text(ks[0],1/9+0.003,"chance",fontsize=5,color="0.4")
    axA.set_xlabel("# subjects"); axA.set_ylabel("balanced accuracy")
    axA.set_title("Decoding learning curve",pad=5)
    for sp in ["top","right"]: axA.spines[sp].set_visible(False); 
    panel_letter(axA,"a",dx=-0.2)
    axB=fig.add_subplot(gs[0,1])
    axB.plot(ks,B_m,"s-",color="#c0392b",ms=4,lw=1.3)
    axB.set_xlabel("# subjects"); axB.set_ylabel("split-half RDM ρ")
    axB.set_title("Geometry stabilization",pad=5); axB.set_ylim(0,1)
    for sp in ["top","right"]: axB.spines[sp].set_visible(False); 
    panel_letter(axB,"b",dx=-0.2)
    axC=fig.add_subplot(gs[0,2])
    axC.errorbar(Ts,C_m,yerr=C_sd,fmt="o-",color="#16a085",ms=4,lw=1.3,capsize=2,elinewidth=0.6)
    axC.set_xlabel("# time samples / trial"); axC.set_ylabel("similarity to full (ρ)")
    axC.set_title("Representation convergence",pad=5); axC.set_ylim(0,1.02)
    for sp in ["top","right"]: axC.spines[sp].set_visible(False); 
    panel_letter(axC,"c",dx=-0.2)
    save(fig, FIG_DIR/"part6"/"Figure_Part6_sample_size"); plt.close(fig)
    print("  LC acc:",[round(a,3) for a in A_m]," | reliab:",[round(b,2) for b in B_m]," | conv:",[round(c,2) for c in C_m])

if __name__=="__main__": main()
