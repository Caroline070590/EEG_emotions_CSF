"""Refit the historically rendered one-second coupling model and check displayed heatmap colors."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from PIL import Image
import matplotlib
from matplotlib.colors import Normalize
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed'))
from part2_synchrony.run_part2 import _zw,_pooled_beta,TOPO,SYNC_COLS
from lib.stats import emotion_slope_lmm
from lib.config import EMOS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def raster_check(number,matrix,cmap,limits):
    p=HERE/'figure4_rasters'/f'panel-{number:03d}.png'
    arr=np.asarray(Image.open(p).convert('RGB'));h,w=arr.shape[:2]
    nr,nc=matrix.shape
    observed=np.array([[arr[int((i+.5)*h/nr),int((j+.5)*w/nc)] for j in range(nc)] for i in range(nr)])
    expected=(matplotlib.colormaps[cmap](Normalize(*limits)(matrix))[:,:,:3]*255).astype(np.uint8)
    expected[~np.isfinite(matrix)]=255
    errors=np.max(np.abs(expected.astype(int)-observed.astype(int)),axis=2)
    return {'raster':str(p),'sha256':sha(p),'n_cells':nr*nc,
            'exact_matching_cells':int((errors==0).sum()),'max_RGB_difference':int(errors.max()),
            'cell_RGB_differences':errors.tolist(),'observed_RGB':observed.tolist(),'expected_RGB':expected.tolist()}

def main():
    start=time.time()
    cache=ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/cache'
    sync=pd.read_csv(cache/'trial_synchrony_metrics.csv')
    topo=pd.read_csv(cache/'trial_metrics_spearman.csv')
    d=_zw(sync.merge(topo,on=['subject','emotion','trial'],validate='one_to_one'),TOPO+SYNC_COLS)
    assert len(d)==3444 and d.subject.nunique()==123
    d.to_csv(HERE/'historical_global_coupling_trial_inputs.csv',index=False)
    pooled=[];slopes=[];inter=[];checks={}
    for k,t in enumerate(TOPO):
        for s in SYNC_COLS:
            beta,p=_pooled_beta(d,t+'_z',s+'_z')
            pooled.append({'topology':t,'synchrony':s,'beta':beta,'p_nominal':p})
            estimates,pi=emotion_slope_lmm(d,t+'_z',s+'_z',ref='neutral')
            inter.append({'topology':t,'synchrony':s,'interaction_p_nominal':pi})
            for emotion,(b,pval) in estimates.items():
                slopes.append({'topology':t,'synchrony':s,'emotion':emotion,'beta':b,'p_nominal':pval})
        print(t,'complete',round(time.time()-start,1),'seconds',flush=True)
    pooled=pd.DataFrame(pooled);slopes=pd.DataFrame(slopes);inter=pd.DataFrame(inter)
    pooled.to_csv(HERE/'historical_global_coupling_pooled_refit.csv',index=False)
    slopes.to_csv(HERE/'historical_global_coupling_emotion_slopes_refit.csv',index=False)
    inter.to_csv(HERE/'historical_global_coupling_interactions_refit.csv',index=False)
    slopes.assign(abs_beta=slopes.beta.abs(),nominal_sig=slopes.p_nominal<.05).groupby('emotion').agg(mean_abs_beta=('abs_beta','mean'),n_nominal_sig=('nominal_sig','sum')).sort_values('mean_abs_beta',ascending=False).to_csv(HERE/'historical_coupling_strength_per_emotion.csv')
    B=pooled.pivot(index='topology',columns='synchrony',values='beta').reindex(index=TOPO,columns=SYNC_COLS).to_numpy()
    checks['pooled']=raster_check(2,B,'PuOr_r',(-.3,.3))
    for k,t in enumerate(TOPO):
        M=slopes[slopes.topology==t].pivot(index='emotion',columns='synchrony',values='beta').reindex(index=EMOS,columns=SYNC_COLS).to_numpy()
        checks[t]=raster_check(4+k,M,'RdBu_r',(-.2,.2))
    heat=sync.groupby(['subject','emotion'])[SYNC_COLS].mean().groupby('emotion').mean().reindex(EMOS)
    z=(heat-heat.mean())/heat.std();checks['fingerprint']=raster_check(0,z.to_numpy(),'magma',(-2,2))
    manifest={'status':'completed numerical-to-rendered comparison','input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [cache/'trial_synchrony_metrics.csv',cache/'trial_metrics_spearman.csv']},
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/part2_synchrony/run_part2.py',ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/lib/stats.py']},
              'n_trial_rows':len(d),'n_pooled':len(pooled),'n_emotion_slopes':len(slopes),
              'pooled_finite_beta':int(np.isfinite(pooled.beta).sum()),'emotion_finite_beta':int(np.isfinite(slopes.beta).sum()),
              'rendered_checks':checks,'elapsed_seconds':time.time()-start,
              'scope':'A color match validates the displayed rounded color-bin estimate, not exact hidden floating-point coefficients. Any mismatch remains explicit.'}
    (HERE/'HISTORICAL_GLOBAL_FIGURE_VALIDATION.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ['n_cells','exact_matching_cells','max_RGB_difference']} for k,v in checks.items()},indent=2),flush=True)

if __name__=='__main__':main()
