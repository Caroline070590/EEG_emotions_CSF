"""Sensor-aligned recurrence multiplex. Exploratory fixed analysis settings."""
import numpy as np
from scipy import signal, spatial, stats
METRICS=['recurrence_rate','determinism','laminarity','mean_diagonal_length','max_diagonal_length','diagonal_entropy']
def runs(x):
    edges=np.diff(np.r_[False,np.asarray(x,bool),False].astype(int))
    return np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
def quantify(r):
    n=len(r); total=r.sum()
    diagonal=np.concatenate([runs(np.diag(r,k)) for k in range(-n+1,n)])
    vertical=np.concatenate([runs(r[:,j]) for j in range(n)])
    dl=diagonal[diagonal>=2];vl=vertical[vertical>=2]
    counts=np.unique(dl,return_counts=True)[1];p=counts/counts.sum() if len(counts) else np.array([])
    return [total/(n*(n-1)),dl.sum()/total if total else 0,vl.sum()/total if total else 0,dl.mean() if len(dl) else 0,dl.max() if len(dl) else 0,-np.sum(p*np.log(p))]
def build(x,rate=.05):
    # Input is a 14-second raw block at 250 Hz. Filter independently before cropping.
    sos=signal.butter(4,[4,45],fs=250,btype='bandpass',output='sos')
    filtered=signal.sosfiltfilt(sos,x,axis=0)
    retained=filtered[500:3000]
    correlation=np.abs(np.corrcoef(stats.rankdata(retained,axis=0),rowvar=False))
    alpha=signal.sosfiltfilt(signal.butter(4,[8,13],fs=250,btype='bandpass',output='sos'),x,axis=0)
    z=signal.hilbert(alpha,axis=0)[500:3000];cross=np.imag(z[:,:,None]*z[:,None,:].conj())
    wpli=np.abs(cross.mean(0))/np.maximum(np.abs(cross).mean(0),1e-12)
    # Anti-aliased 125-Hz series; m=3,tau=2 (16 ms), anchors every 5 samples (40 ms).
    small=signal.resample_poly(filtered,1,2,axis=0)[250:1500]
    small=(small-small.mean(0))/np.maximum(small.std(0),1e-12)
    anchors=np.arange(0,len(small)-4,5)
    embedded=np.stack([small[anchors+j] for j in [0,2,4]],axis=-1)
    n=len(anchors);eligible=np.abs(anchors[:,None]-anchors[None,:])>12 # exclude <=96 ms
    upper=np.triu(eligible,1);patterns=[];metrics=[]
    for c in range(x.shape[1]):
        dist=spatial.distance.squareform(spatial.distance.pdist(embedded[:,c,:]))
        threshold=np.quantile(dist[upper],rate)
        r=(dist<=threshold)&eligible
        patterns.append(r[upper]);metrics.append(quantify(r))
    patterns=np.asarray(patterns,float);joint=patterns@patterns.T
    denom=patterns.sum(1);recurrence=joint/np.maximum(denom[:,None]+denom[None,:]-joint,1)
    graphs=np.stack([correlation,wpli,recurrence])
    for g in graphs:np.fill_diagonal(g,0)
    return graphs.astype('float32'),np.asarray(metrics,dtype='float32')
