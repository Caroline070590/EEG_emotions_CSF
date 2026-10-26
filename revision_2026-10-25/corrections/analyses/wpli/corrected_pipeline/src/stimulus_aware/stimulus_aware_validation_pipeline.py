#!/usr/bin/env python3
"""Stimulus-aware sensitivity analysis for the FACED trial-level EEG export.

This script reproduces the analyses used for Supplementary Figure S21.
It deliberately does NOT recompute QG because the current manuscript does not
specify the maximum lag searched when selecting the trial-specific QG lag.
No undocumented value is inserted.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

from pathlib import Path
import argparse, itertools, json, re, tempfile, zipfile
import numpy as np
import pandas as pd
from scipy import signal, stats
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from statsmodels.stats.multitest import multipletests

CH=['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4','T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']
EMOTIONS=['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']
E={e:i for i,e in enumerate(EMOTIONS)}
VE={}
for e,a,b in [('anger',1,3),('disgust',4,6),('fear',7,9),('sadness',10,12),('neutral',13,16),('amusement',17,19),('inspiration',20,22),('joy',23,25),('tenderness',26,28)]:
    for v in range(a,b+1): VE[v]=e
VG={e:[v for v in range(1,29) if VE[v]==e] for e in EMOTIONS}
FS=250; IU=np.triu_indices(30,1)
SOS=signal.butter(4,[1,45],btype='bandpass',fs=FS,output='sos')

def spearman_vec(X):
    R=np.apply_along_axis(stats.rankdata,1,X)
    A=np.corrcoef(R); A=np.nan_to_num(np.abs(A),nan=0.0,posinf=0.0,neginf=0.0); np.fill_diagonal(A,0)
    return A[IU].astype(np.float32)

def mtf_vec(X,q=30):
    n=X.shape[1]; counts=np.zeros((q,q),float)
    for x in X:
        r=stats.rankdata(x,method='average'); s=np.floor((r-1)*q/n).astype(int); s=np.clip(s,0,q-1)
        np.add.at(counts,(s[:-1],s[1:]),1)
    np.fill_diagonal(counts,0); rows=counts.sum(1,keepdims=True)
    P=np.divide(counts,rows,out=np.zeros_like(counts),where=rows>0); A=(P+P.T)/2; np.fill_diagonal(A,0)
    return A[IU].astype(np.float32)

def synchronization(X):
    U=signal.sosfiltfilt(SOS,X,axis=1); Z=signal.hilbert(U,axis=1); amp=np.abs(Z)
    unit=Z/np.maximum(amp,1e-15); plv=np.abs(unit@unit.conj().T/X.shape[1])
    im=np.imag(Z[:,None,:]*np.conj(Z[None,:,:])); pli=np.abs(np.mean(np.sign(im),axis=2))
    den=np.mean(np.abs(im),axis=2); wpli=np.divide(np.abs(np.mean(im,axis=2)),den,out=np.zeros_like(den),where=den>0)
    aec=np.corrcoef(amp)
    sd=U.std(1,keepdims=True); sd[sd==0]=1; Uz=(U-U.mean(1,keepdims=True))/sd
    N=X.shape[1]; nfft=512; F=np.fft.fft(Uz,nfft,axis=1)
    circ=np.fft.ifft(F[:,None,:]*np.conj(F[None,:,:]),axis=2).real
    full=np.concatenate([circ[:,:,-(N-1):],circ[:,:,:N]],axis=2)/N
    idx=np.argmax(np.abs(full),axis=2); maxxc=np.take_along_axis(np.abs(full),idx[:,:,None],axis=2)[:,:,0]
    lag_axis=np.arange(-(N-1),N); lag=np.abs(lag_axis[idx])*1000/FS
    return np.array([np.nanmean(m[IU]) for m in [plv,pli,wpli,aec,maxxc,lag]],dtype=np.float32)

def one_file(path):
    p=Path(path); v=int(re.search(r'trial_(\d+)\.csv$',p.name).group(1)); folder=p.parts[-3].lower()
    d=pd.read_csv(p,usecols=['subject_id','trial_id','emotion_label','emotion_name']+CH)
    sid=int(d.subject_id.iloc[0]); tid=int(d.trial_id.iloc[0]); emo=str(d.emotion_name.iloc[0]).lower()
    if folder!=emo or VE[v]!=emo or tid+1!=v: raise ValueError(f'Stimulus mapping mismatch: {p}')
    X=d[CH].to_numpy(float).T
    if X.shape!=(30,250): raise ValueError(f'Expected 30x250 EEG, got {X.shape}: {p}')
    return sid,v,E[emo],spearman_vec(X),mtf_vec(X),synchronization(X)

def locate_raw(root):
    cands=list(root.rglob('trial_001.csv'))
    if not cands: raise FileNotFoundError('No trial_001.csv found after extraction')
    # .../<EEG-data-raw>/<emotion>/<PatientXXX>/trial_001.csv
    return cands[0].parents[2]

def extract_features(zip_path,out,n_jobs):
    with tempfile.TemporaryDirectory(prefix='faced_stimulus_') as td:
        td=Path(td)
        with zipfile.ZipFile(zip_path) as zf: zf.extractall(td)
        raw=locate_raw(td); files=sorted(raw.glob('**/trial_*.csv'))
        if len(files)!=3444: raise RuntimeError(f'Expected 3444 trial files, got {len(files)}')
        res=Parallel(n_jobs=n_jobs,backend='loky',verbose=5)(delayed(one_file)(str(f)) for f in files)
    subject=np.array([r[0] for r in res],np.int16); video=np.array([r[1] for r in res],np.int8); label=np.array([r[2] for r in res],np.int8)
    sp=np.stack([r[3] for r in res]); mtf=np.stack([r[4] for r in res]); sync=np.stack([r[5] for r in res])
    order=np.lexsort((video,subject)); subject,video,label,sp,mtf,sync=[x[order] for x in (subject,video,label,sp,mtf,sync)]
    assert len(np.unique(subject))==123 and len(np.unique(video))==28
    assert (pd.crosstab(subject,video).values==1).all()
    np.savez_compressed(out/'trial_features.npz',subject=subject,video=video,label=label,spearman=sp,mtf=mtf,sync=sync)
    meta=pd.DataFrame({'subject_id':subject,'video_id':video,'emotion':[EMOTIONS[i] for i in label]})
    for j,n in enumerate(['PLV','PLI','wPLI','AEC','MaxXC','Lag']): meta[n]=sync[:,j]
    meta.to_csv(out/'trial_scalar_synchronization.csv',index=False)
    return subject,video,label,sp,mtf,sync,meta

def representation_validation(video,reps,out,seed,B=10000):
    labs=np.array([E[VE[v]] for v in range(1,29)]); tri=np.triu_indices(28,1); rng=np.random.default_rng(seed)
    summaries=[]; retrieval=[]
    for rep,X in reps.items():
        V=np.stack([X[video==v].mean(0) for v in range(1,29)]); S=np.corrcoef(V); sv=S[tri]; same=labs[tri[0]]==labs[tri[1]]
        within=float(sv[same].mean()); between=float(sv[~same].mean()); obs=within-between
        null=[]
        for _ in range(B):
            lp=rng.permutation(labs); sm=lp[tri[0]]==lp[tri[1]]; null.append(sv[sm].mean()-sv[~sm].mean())
        p=(1+np.sum(np.asarray(null)>=obs))/(B+1)
        def retrieve(lp):
            pp=[]; margins=[]; true_scores=[]
            for i in range(28):
                scores=[]
                for e in range(9):
                    inds=np.where(lp==e)[0]; inds=inds[inds!=i]; scores.append(float(S[i,inds].mean()) if len(inds) else -np.inf)
                scores=np.asarray(scores); t=int(lp[i]); pp.append(int(np.argmax(scores))); margins.append(scores[t]-np.max(np.delete(scores,t))); true_scores.append(scores[t])
            return np.asarray(pp),np.asarray(margins),np.asarray(true_scores)
        pred,marg,true_score=retrieve(labs); acc=float((pred==labs).mean()); nullacc=[]
        for _ in range(B):
            lp=rng.permutation(labs); pp,_,_=retrieve(lp); nullacc.append((pp==lp).mean())
        pacc=(1+np.sum(np.asarray(nullacc)>=acc))/(B+1)
        summaries.append({'representation':rep,'same_emotion_mean_similarity':within,'different_emotion_mean_similarity':between,'same_minus_different':obs,'permutation_p':p,'retrieval_accuracy':acc,'retrieval_permutation_p':pacc,'mean_true_emotion_margin':float(marg.mean()),'n_stimuli':28,'n_permutations':B})
        for i,v in enumerate(range(1,29)): retrieval.append({'representation':rep,'video_id':v,'emotion':VE[v],'predicted_emotion':EMOTIONS[pred[i]],'correct':pred[i]==labs[i],'true_emotion_margin':marg[i],'true_emotion_mean_similarity':true_score[i]})
        pd.DataFrame(S,index=range(1,29),columns=range(1,29)).to_csv(out/f'{rep.lower()}_stimulus_similarity.csv')
    pd.DataFrame(summaries).to_csv(out/'cross_stimulus_similarity_summary.csv',index=False); pd.DataFrame(retrieval).to_csv(out/'leave_one_stimulus_retrieval.csv',index=False)

def cross_stimulus_decode(video,label,reps,out):
    rows=[]; cms=[]
    for rep,X in reps.items():
        for fold in range(3):
            held=[VG[e][fold] for e in EMOTIONS]; test=np.isin(video,held); train=~test
            scaler=StandardScaler().fit(X[train]); Xt=scaler.transform(X[train]); Xv=scaler.transform(X[test])
            clf=LogisticRegression(max_iter=1500,C=1.0,solver='lbfgs'); clf.fit(Xt,label[train]); yp=clf.predict(Xv)
            rows.append({'representation':rep,'fold':fold+1,'held_out_videos':','.join(map(str,held)),'n_train':int(train.sum()),'n_test':int(test.sum()),'accuracy':accuracy_score(label[test],yp),'balanced_accuracy':balanced_accuracy_score(label[test],yp)})
            cm=confusion_matrix(label[test],yp,labels=np.arange(9),normalize='true')
            for i,a in enumerate(EMOTIONS):
                for j,b in enumerate(EMOTIONS): cms.append({'representation':rep,'fold':fold+1,'true_emotion':a,'predicted_emotion':b,'proportion':cm[i,j]})
    pd.DataFrame(rows).to_csv(out/'cross_stimulus_decoding.csv',index=False); pd.DataFrame(cms).to_csv(out/'cross_stimulus_decoding_confusion.csv',index=False)

def sync_sensitivity(meta,out,seed):
    df=meta.copy()
    for m in ['PLV','PLI','wPLI','AEC','MaxXC','Lag']:
        df[m+'_z']=df.groupby('subject_id')[m].transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else 0.0)
    def eff(m,e,vt=None,vn=None):
        dd=df
        if vt is not None: dd=dd[~((dd.emotion==e)&(dd.video_id==vt))]
        if vn is not None: dd=dd[~((dd.emotion=='neutral')&(dd.video_id==vn))]
        x=dd[dd.emotion==e].groupby('subject_id')[m+'_z'].mean(); n=dd[dd.emotion=='neutral'].groupby('subject_id')[m+'_z'].mean(); d=(x-n).dropna(); return float(d.mean()),d.to_numpy()
    allr=[]
    for m in ['PLV','PLI','wPLI','AEC','MaxXC','Lag']:
        for e in [x for x in EMOTIONS if x!='neutral']:
            est,d=eff(m,e); t,p=stats.ttest_1samp(d,0); allr.append({'metric':m,'emotion':e,'effect_beta_like':est,'paired_t':t,'paired_p':p})
    a=pd.DataFrame(allr); a['q_BH_48']=multipletests(a.paired_p,method='fdr_bh')[1]; a.to_csv(out/'synchronization_contrast_reproduction_check.csv',index=False)
    headline=[('PLV','disgust'),('PLV','anger'),('PLV','fear'),('AEC','disgust'),('MaxXC','disgust'),('MaxXC','anger'),('Lag','fear'),('Lag','disgust'),('Lag','joy')]
    rng=np.random.default_rng(seed); rows=[]; folds=[]; exact=[]
    for m,e in headline:
        full,d=eff(m,e); boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)]); lo,hi=np.quantile(boot,[.025,.975]); vals=[]
        for vt in VG[e]:
            for vn in VG['neutral']:
                est,_=eff(m,e,vt,vn); vals.append(est); folds.append({'metric':m,'emotion':e,'drop_target_video':vt,'drop_neutral_video':vn,'effect_beta_like':est,'same_direction_as_full':np.sign(est)==np.sign(full)})
        vals=np.asarray(vals); rows.append({'metric':m,'emotion':e,'full_effect_beta_like':full,'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'leave_one_each_min':vals.min(),'leave_one_each_max':vals.max(),'sign_stability_fraction':np.mean(np.sign(vals)==np.sign(full)),'n_leave_one_each_folds':len(vals)})
        vids=VG[e]+VG['neutral']; vmean={v:df[df.video_id==v][m+'_z'].mean() for v in vids}; obs=np.mean([vmean[v] for v in VG[e]])-np.mean([vmean[v] for v in VG['neutral']]); ds=[]
        for comb in itertools.combinations(vids,3):
            A=set(comb); B=[v for v in vids if v not in A]; ds.append(np.mean([vmean[v] for v in A])-np.mean([vmean[v] for v in B]))
        ds=np.asarray(ds); exact.append({'metric':m,'emotion':e,'stimulus_level_effect':obs,'exact_two_sided_p':np.mean(np.abs(ds)>=abs(obs)-1e-15),'exact_directional_p':np.mean(ds<=obs+1e-15) if obs<0 else np.mean(ds>=obs-1e-15),'n_exact_allocations':len(ds)})
    pd.DataFrame(rows).to_csv(out/'headline_synchronization_stimulus_robustness.csv',index=False); pd.DataFrame(folds).to_csv(out/'headline_synchronization_stimulus_deletion_folds.csv',index=False)
    ex=pd.DataFrame(exact); ex['q_BH_9_two_sided']=multipletests(ex.exact_two_sided_p,method='fdr_bh')[1]; ex.to_csv(out/'headline_synchronization_exact_stimulus_permutation.csv',index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n-jobs',type=int,default=8); ap.add_argument('--seed',type=int,default=20260810); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    subject,video,label,sp,mtf,sync,meta=extract_features(Path(args.zip),out,args.n_jobs)
    reps={'Spearman':sp,'MTF':mtf}; representation_validation(video,reps,out,args.seed); cross_stimulus_decode(video,label,reps,out); sync_sensitivity(meta,out,args.seed)
    pd.DataFrame({'video_id':range(1,29),'emotion':[VE[v] for v in range(1,29)]}).to_csv(out/'stimulus_design_map.csv',index=False)
    with open(out/'analysis_manifest.json','w') as f: json.dump({'n_subjects':123,'n_stimuli':28,'n_trials':3444,'n_channels':30,'samples_per_trial':250,'seed':args.seed,'QG':'not recomputed because maximum lag-search range is not stated in the current manuscript','self_report':'not run because ratings are not present in the uploaded EEG ZIP'},f,indent=2)
    print('Done. Summary files written to',out)
if __name__=='__main__': main()
