"""Analyses, not historical figure reproduction."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import gudhi
import networkx as nx
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.stats.multitest import multipletests

FS = 250
CHANNELS = ['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ',
            'C3','C4','T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8',
            'PO3','PO4','OZ','O1','O2']
EMOTIONS = ['anger','disgust','fear','sadness','neutral','amusement',
            'inspiration','joy','tenderness']
VIDEO_EMOTION = dict(zip(range(28), np.repeat(EMOTIONS, [3,3,3,3,4,3,3,3,3])))
BANDS = {'theta': (4,8), 'alpha': (8,13), 'beta': (13,30), 'gamma': (30,45),
         'broadband': (1,45), 'restricted': (4,45)}
PRIMARY_BANDS = ['theta','alpha','beta','gamma']
# A geometrically defined sensitivity set, NOT an EMG detector or cleaned EEG.
PERIPHERAL = ['FP1','FP2','F7','F8','T7','T8','P7','P8']
GROUPS = {'all': np.arange(30),
          'core': np.array([i for i,c in enumerate(CHANNELS) if c not in PERIPHERAL]),
          'peripheral': np.array([CHANNELS.index(c) for c in PERIPHERAL])}
SYNC = ['plv','pli','wpli','aec','maxxc','lag_ms']
TOPOLOGY = ['transitivity','efficiency','path_length','gamma','lambda','sigma']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_trial(path, expected_samples):
    d = pd.read_csv(path)
    for col in ['subject_id','trial_id','emotion_name']:
        if d[col].nunique(dropna=False) != 1:
            raise ValueError(f'Nonconstant {col}: {path}')
    subject, raw_trial = int(d.subject_id.iloc[0]), int(d.trial_id.iloc[0])
    # Both supplied CSV exporters store local zero-based video IDs in trial_id.
    # sample_index is a different identifier and is never used as a video ID.
    if not 0<=raw_trial<28:
        raise ValueError(f'Expected local zero-based video ID: {path}')
    video=raw_trial
    if 'window_config' in d:
        if d.window_config.nunique()!=1 or d.window_config.iloc[0]!='full_trial' or not 0<=raw_trial<28:
            raise ValueError(f'Invalid full-trial identity: {path}')
    emotion = str(d.emotion_name.iloc[0]).lower()
    if VIDEO_EMOTION[video] != emotion:
        raise ValueError(f'Video/emotion disagreement: {path}')
    X = d[CHANNELS].to_numpy(float)
    if X.shape != (expected_samples,30) or not np.isfinite(X).all():
        raise ValueError(f'Invalid EEG shape/values: {path}, {X.shape}')
    if np.any(X.std(0) == 0):
        raise ValueError(f'Constant channel: {path}')
    return X, dict(subject=subject, video=video, emotion=emotion)


def validate_balanced(meta, expected_subjects=None):
    if meta.duplicated(['subject','video']).any():
        raise ValueError('Duplicate participant/video keys')
    if expected_subjects is not None and meta.subject.nunique() != expected_subjects:
        raise ValueError('Unexpected participant count')
    if any(set(g.video) != set(range(28)) for _,g in meta.groupby('subject')):
        raise ValueError('Each participant must have all 28 videos')
    if any(VIDEO_EMOTION[int(v)] != e for v,e in zip(meta.video,meta.emotion)):
        raise ValueError('Video/category mapping mismatch')


def signed_spearman(X):
    R = np.corrcoef(stats.rankdata(X, axis=0), rowvar=False)
    return np.clip(R, -1, 1)


def transition(X, q, lag):
    S = np.empty(X.shape, dtype=np.int16)
    for c in range(X.shape[1]):
        edges = np.unique(np.quantile(X[:,c], np.linspace(0,1,q+1)))
        S[:,c] = np.digitize(X[:,c], edges[1:-1])
    counts = np.bincount((S[:-lag]*q + S[lag:]).ravel(), minlength=q*q).reshape(q,q).astype(float)
    np.fill_diagonal(counts,0)
    total = counts.sum(1,keepdims=True)
    P = np.divide(counts,total,out=np.zeros_like(counts),where=total>0)
    return (P+P.T)/2


def qg_lag(X):
    """Historical mean-channel lag, with an explicit near-zero-mean diagnostic."""
    x = X.mean(1); x = x-x.mean()
    ratio = np.sqrt(np.mean(x*x))/np.sqrt(np.mean((X-X.mean(0))**2))
    max_lag = min(len(X)//4,len(X)-2)
    denom = x@x
    if denom == 0:
        return 1,ratio,True
    f = np.fft.rfft(x,n=2*len(x))
    acf = np.fft.irfft(f*f.conj(),n=2*len(x))[1:max_lag+1]/denom
    found = np.flatnonzero(np.abs(acf)<=.1)
    k = int(found[0]+1) if len(found) else int(np.argmin(np.abs(acf))+1)
    return k,ratio,ratio<1e-8


def threshold(A, density):
    A = np.array(A, dtype=float, copy=True)
    np.fill_diagonal(A,0)
    if not np.allclose(A,A.T) or np.any(A<0):
        raise ValueError('Threshold expects symmetric nonnegative weights')
    w = A[np.triu_indices(len(A),1)]
    positive = w[w>0]
    target = min(len(positive),max(1,round(density*len(w))))
    if not len(positive):
        return np.zeros_like(A,dtype=bool)
    return (A>=np.sort(positive)[-target]) & (A>0)


def graph_metrics(A,density=.1,n_random=20,seed=0):
    B = threshold(A,density); G = nx.from_numpy_array(B)
    component = G.subgraph(max(nx.connected_components(G),key=len))
    L = nx.average_shortest_path_length(component) if len(component)>1 else np.nan
    C = nx.average_clustering(G)
    result = dict(transitivity=nx.transitivity(G),efficiency=nx.global_efficiency(G),
                  path_length=L,realized_density=nx.density(G),gcc_nodes=len(component),
                  n_edges=G.number_of_edges(),n_nodes=len(G))
    cr,lr = [],[]; rng=np.random.default_rng(seed)
    for _ in range(n_random):
        H=G.copy(); swaps=5*H.number_of_edges()
        if swaps == 0:
            continue
        try:
            nx.double_edge_swap(H,nswap=swaps,max_tries=max(100,20*swaps),
                                seed=int(rng.integers(2**31-1)))
        except (nx.NetworkXError,nx.NetworkXAlgorithmError):
            continue
        cc=H.subgraph(max(nx.connected_components(H),key=len))
        cr.append(nx.average_clustering(H))
        lr.append(nx.average_shortest_path_length(cc) if len(cc)>1 else np.nan)
    gamma = C/np.mean(cr) if cr and np.mean(cr)>0 else np.nan
    lam = L/np.mean(lr) if lr and np.mean(lr)>0 else np.nan
    result.update(gamma=gamma, **{'lambda':lam}, sigma=gamma/lam if lam>0 else np.nan,
                  randomizations_successful=len(cr),randomizations_requested=n_random)
    return result


def analytic(X,band):
    sos=signal.butter(4,BANDS[band],btype='bandpass',fs=FS,output='sos')
    filtered=signal.sosfiltfilt(sos,X,axis=0)
    return filtered,signal.hilbert(filtered,axis=0)


def synchrony(filtered,z,include_crosscorr=True):
    n=len(z); iu=np.triu_indices(z.shape[1],1)
    u=z/np.maximum(np.abs(z),np.finfo(float).tiny)
    # Full analytic signals retain amplitude weights in wPLI.
    cross=z[:,iu[0]]*z[:,iu[1]].conj(); im=cross.imag
    den=np.mean(np.abs(im),0)
    wpli=np.divide(np.abs(im.mean(0)),den,out=np.zeros_like(den),where=den>0)
    out=dict(plv=np.abs((u.conj().T@u)/n)[iu].mean(),
             pli=np.abs(np.sign(im).mean(0)).mean(),wpli=wpli.mean(),
             aec=np.corrcoef(np.abs(z),rowvar=False)[iu].mean())
    if include_crosscorr:
        X=(filtered-filtered.mean(0))/filtered.std(0)
        F=np.fft.rfft(X,n=2*n,axis=0)
        cc=np.fft.irfft(F[:,iu[0]]*F[:,iu[1]].conj(),n=2*n,axis=0)/n
        cc=np.concatenate([cc[-(n-1):],cc[:n]],axis=0)
        where=np.abs(cc).argmax(0)
        out.update(maxxc=np.abs(cc[where,np.arange(len(where))]).mean(),
                   lag_ms=(np.abs(where-(n-1))*1000/FS).mean())
    return {k:float(v) for k,v in out.items()}


def bh(values):
    p=np.asarray(values,float); good=np.isfinite(p); q=np.full(len(p),np.nan)
    if good.any():
        # Missing tests still count toward the declared family; they are not discoveries.
        q_all=multipletests(np.where(good,p,1.),method='fdr_bh')[1]
        q[good]=q_all[good]
    return q


def exact_video_contrasts(frame,metric):
    """35 label allocations; video, not participant or epoch, is inference unit."""
    wide=frame.pivot(index='subject',columns='video',values=metric).sort_index(axis=1)
    if wide.shape[1]!=28 or wide.isna().any().any():
        raise ValueError(f'Incomplete participant/video grid for {metric}')
    video_means=wide.mean(0)
    neutral=[v for v in range(28) if VIDEO_EMOTION[v]=='neutral']
    rows=[]
    for emotion in EMOTIONS:
        if emotion=='neutral': continue
        target=[v for v in range(28) if VIDEO_EMOTION[v]==emotion]
        vals=video_means.loc[target+neutral].to_numpy()
        observed=vals[:3].mean()-vals[3:].mean()
        null=[]
        for comb in itertools.combinations(range(7),3):
            mask=np.zeros(7,bool);mask[list(comb)]=True
            null.append(vals[mask].mean()-vals[~mask].mean())
        p=np.mean(np.abs(null)>=abs(observed)-1e-12)
        diff=wide[target].mean(1)-wide[neutral].mean(1)
        se=diff.std(ddof=1)/np.sqrt(len(diff))
        margin=stats.t.ppf(.975,len(diff)-1)*se if len(diff)>1 else np.nan
        rows.append(dict(emotion=emotion,metric=metric,effect_raw=observed,
                         participant_ci_low=observed-margin,participant_ci_high=observed+margin,
                         participant_dz=observed/diff.std(ddof=1) if diff.std(ddof=1)>0 else np.nan,
                         p=p,n_participants=len(wide),n_target_videos=3,n_neutral_videos=4,
                         n_allocations=35,inference='exact_video_absolute_mean_difference'))
    return rows


def flag_persistence(A):
    """Weighted flag filtration, edge birth 1-A; no metric-space claim."""
    st=gudhi.SimplexTree(); n=len(A)
    for i in range(n): st.insert([i],filtration=0.)
    for i,j in zip(*np.triu_indices(n,1)):
        st.insert([int(i),int(j)],filtration=float(1-A[i,j]))
    st.expansion(2);st.persistence()
    out={}
    for dim in (0,1):
        intervals=st.persistence_intervals_in_dimension(dim)
        finite=intervals[np.isfinite(intervals).all(1)]
        life=finite[:,1]-finite[:,0]
        positive=life[life>0]; total=positive.sum()
        out.update({f'h{dim}_count':len(positive),f'h{dim}_total':total,
                    f'h{dim}_max':positive.max() if len(positive) else 0.,
                    f'h{dim}_entropy':stats.entropy(positive/total) if total else 0.})
    return out
