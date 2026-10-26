"""Verify retained one-second wPLI caches against every locally available raw trial.

Writes only beside this script. Estimator audit retains both the historical
phase-only and standard amplitude-weighted values; no inference is selected.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfiltfilt, hilbert
from scipy.stats import rankdata
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / 'EEG-data-raw'
CH = ['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4',
      'T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']
REGIONS = {'frontal':['FP1','FP2','F3','F4','F7','F8','FZ'],
           'central':['FC1','FC2','FC5','FC6','CZ','C3','C4'], 'temporal':['T7','T8'],
           'parietal':['CP1','CP2','CP5','CP6','P3','P4','P7','P8','PZ'],
           'occipital':['PO3','PO4','O1','O2','OZ']}
I,J = np.triu_indices(30,1)
MASKS = {r: np.isin(I,[CH.index(c) for c in cs]) & np.isin(J,[CH.index(c) for c in cs]) for r,cs in REGIONS.items()}
BANDS = {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45),
         'broadband_1_45':(1,45),'broadband_4_45':(4,45)}
B,A = butter(4,[1/125,45/125],btype='band')
SOS = butter(4,[1,45],btype='band',fs=250,output='sos')
METRICS = ['plv','pli','wpli','amp_env_corr','max_cross_corr','lag_ms']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_metrics(u):
    z=hilbert(u,axis=0); unit=np.exp(1j*np.angle(z)); amp=np.abs(z)
    cross=z[:,I]*z[:,J].conj(); im=cross.imag
    phase_im=(unit[:,I]*unit[:,J].conj()).imag
    def wpli(v):
        den=np.abs(v).mean(0)
        return np.divide(np.abs(v.mean(0)),den,out=np.zeros_like(den),where=den>0)
    env=(amp-amp.mean(0))/(amp.std(0)+1e-12)
    uz=(u-u.mean(0))/(u.std(0)+1e-12);n=len(u)
    f=np.fft.rfft(uz,n=2*n,axis=0)
    circ=np.fft.irfft(f[:,I]*f[:,J].conj(),n=2*n,axis=0)/n
    cc=np.concatenate([circ[-(n-1):],circ[:n]],axis=0)
    loc=np.abs(cc).argmax(0)
    return {'plv':np.abs((unit[:,I]*unit[:,J].conj()).mean(0)),
            'pli':np.abs(np.sign(im).mean(0)), 'wpli':wpli(im),
            'amp_env_corr':(env[:,I]*env[:,J]).mean(0),
            'max_cross_corr':np.abs(cc[loc,np.arange(len(I))]),
            'lag_ms':np.abs(loc-(n-1))*4., 'phase_only_wpli':wpli(phase_im)}


def one(path, sample):
    d=pd.read_csv(path)
    sid=int(d.subject_id.iloc[0]);video=int(d.trial_id.iloc[0])+1
    assert d.subject_id.nunique()==d.trial_id.nunique()==d.emotion_name.nunique()==1
    assert int(re.search(r'Patient(\d+)',path.parent.name).group(1))==sid+1
    assert int(re.search(r'trial_(\d+)',path.name).group(1))==video
    emotion=str(d.emotion_name.iloc[0]);assert path.parent.parent.name==emotion
    x=d[CH].to_numpy(float);assert x.shape==(250,30) and np.isfinite(x).all()
    u=filtfilt(B,A,x,axis=0);m=pair_metrics(u)
    base={'subject_id':sid,'video_id':video,'subject':sid+1,'trial':path.stem,'emotion':emotion}
    global_row={**base,**{k:float(v.mean()) for k,v in m.items()}}
    # Independent SOS implementation, matching the behavioral/stimulus generators.
    zs=hilbert(sosfiltfilt(SOS,x,axis=0),axis=0)
    im=(zs[:,I]*zs[:,J].conj()).imag;den=np.abs(im).mean(0)
    global_row['wpli_sos']=float(np.divide(np.abs(im.mean(0)),den,out=np.zeros_like(den),where=den>0).mean())
    ranks=np.apply_along_axis(rankdata,0,x);corr=np.abs(np.corrcoef(ranks.T));np.fill_diagonal(corr,0)
    regional=[{**base,'region':r,'n_channels':len(REGIONS[r]),
               **{k:float(v[mask].mean()) for k,v in m.items()},
               'nodal_strength':float(corr.sum(1)[[CH.index(c) for c in REGIONS[r]]].mean())}
              for r,mask in MASKS.items()]
    bands=[]
    if sample:
        for band,(low,high) in BANDS.items():
            b,a=butter(4,[low/125,high/125],btype='band')
            v=pair_metrics(filtfilt(b,a,x,axis=0))
            bands.append({**base,'band':band,**{k:float(val.mean()) for k,val in v.items()}})
    return global_row,regional,bands,{'path':str(path.relative_to(ROOT)),'sha256':sha(path)}


def comparison(reference, recomputed, metric, candidate='wpli'):
    a=reference.set_index(['subject_id','video_id']).sort_index()
    b=recomputed.set_index(['subject_id','video_id']).sort_index()
    assert a.index.equals(b.index)
    err=np.abs(a[metric].to_numpy()-b[candidate].to_numpy())
    return {'n':len(err),'max_absolute_error':float(err.max()),'mean_absolute_error':float(err.mean()),
            'matches_at_1e-6':bool(np.all(err<1e-6))}


def main():
    start=time.time();files=sorted(RAW.glob('*/*/trial_*.csv'));assert len(files)==3444,len(files)
    rng=np.random.default_rng(20261025)
    sample=set(rng.choice(len(files),size=63,replace=False))
    out=Parallel(n_jobs=4,backend='loky',verbose=5)(delayed(one)(p,i in sample) for i,p in enumerate(files))
    glob=pd.DataFrame([v[0] for v in out]).sort_values(['subject_id','video_id'])
    regional=pd.DataFrame([r for v in out for r in v[1]]).sort_values(['subject_id','video_id','region'])
    bands=pd.DataFrame([r for v in out for r in v[2]])
    assert not glob.duplicated(['subject_id','video_id']).any()
    assert (pd.crosstab(glob.subject_id,glob.video_id).values==1).all()
    glob.to_csv(HERE/'raw_recomputed_global.csv',index=False)
    regional.to_csv(HERE/'raw_recomputed_regional.csv',index=False)
    bands.to_csv(HERE/'raw_recomputed_band_sample.csv',index=False)
    canonical=glob[['subject_id','video_id','emotion']+METRICS].rename(columns=dict(zip(METRICS,['PLV','PLI','wPLI','AEC','MaxXC','Lag'])))
    canonical.to_csv(HERE/'trial_scalar_synchronization_standard_recomputed.csv',index=False)
    (HERE/'RAW_INPUT_SHA256.json').write_text(json.dumps([v[3] for v in out],indent=2)+'\n')
    caches={
      'retained_global':'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/cache/trial_synchrony_metrics.csv',
      'historical_phase_only':'emotions-test4/emotion_eeg_paper/cache/trial_synchrony_metrics.csv',
      'released_derived':'EEG_emotions_CSF_release/faced_1s_exploratory/derived/trial_scalar_synchronization.csv',
      'behavioral_verified':'EEG_emotion_final_code_organized_2026-09-04/appendix_g/verification_rerun_2026-09-04/trial_scalar_synchronization.csv',
      'stimulus_aware':'Stimulus_aware_FACED_S21_package/results/trial_scalar_synchronization.csv'}
    checks={}
    for name,relative in caches.items():
        p=ROOT/relative;d=pd.read_csv(p)
        if 'subject_id' not in d:
            d['subject_id']=d.subject-1;d['video_id']=d.trial.str.extract(r'(\d+)')[0].astype(int)
        metric='wPLI' if 'wPLI' in d else 'wpli'
        checks[name]={'path':relative,'sha256':sha(p),
                      'standard_ba':comparison(d,glob,metric),'standard_sos':comparison(d,glob,metric,'wpli_sos'),
                      'phase_only':comparison(d,glob,metric,'phase_only_wpli')}
    bp=ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/band_robustness_results_and_code/emotion_eeg_paper_fixed/cache/trial_synchrony_bands.csv'
    bc=pd.read_csv(bp).set_index(['subject','trial'])
    band_checks=[]
    for _,r in bands.iterrows():
        old=bc.loc[(r.subject,r.trial),r.band+'__wpli']
        band_checks.append({'subject_id':r.subject_id,'video_id':r.video_id,'band':r.band,'saved_wpli':old,
                            'standard_wpli':r.wpli,'phase_only_wpli':r.phase_only_wpli,'absolute_error':abs(old-r.wpli)})
    pd.DataFrame(band_checks).to_csv(HERE/'band_cache_verification.csv',index=False)
    manifest={'status':'raw comparisons completed','n_raw_trials':len(glob),'n_regional_rows':len(regional),
              'band_sample_trials':63,'band_sample_estimates':len(band_checks),
              'band_cache_sha256':sha(bp),'band_max_abs_error':float(pd.DataFrame(band_checks).absolute_error.max()),
              'cache_checks':checks,'elapsed_seconds':time.time()-start,'script_sha256':sha(__file__),
              'raw_manifest_sha256':sha(HERE/'RAW_INPUT_SHA256.json'),
              'scope':'Full broadband wPLI raw recomputation; band cache sampled across 63 trials and seven bands; regional estimates regenerated for downstream verification. Does not certify all figure generators.'}
    (HERE/'RAW_LINEAGE_VALIDATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
