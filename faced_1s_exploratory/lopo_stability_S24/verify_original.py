"""Step 0: reproduce the July 2026 S24 outputs (all 5,166 rows: F, p, q) from the July input caches."""
import sys,numpy as np,pandas as pd
from pathlib import Path
from statsmodels.stats.multitest import multipletests
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lopo_lib import rm_anova
C=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'original_2026-07'
L=pd.read_csv(C/'LOSO_headline_effect_stability_long.csv')
src={('topology','mtf'):'trial_metrics_mtf',('topology','qg'):'trial_metrics_qg',('topology','spearman'):'trial_metrics_spearman',
     ('synchrony','all_pairs'):'trial_synchrony_metrics',('higher_order_topology','mtf'):'tda_mtf',('higher_order_topology','qg'):'tda_qg',('higher_order_topology','spearman'):'tda_spearman'}
cells={}
for (dom,meth),f in src.items():
    d=pd.read_csv(C/'cache'/f'{f}.csv').replace([np.inf,-np.inf],np.nan)
    d['subject']=d.subject.astype(str).str.replace('Patient','').astype(int)
    for meas in L[(L.domain==dom)&(L.method==meth)].measure.unique():
        cells[(dom,meth,meas)]=d.groupby(['subject','emotion'])[meas].mean().unstack()
rows=[]
for s in sorted(L.left_out_subject.unique()):
    for k,cell in cells.items():
        r=rm_anova(cell.drop(index=s));rows.append(dict(domain=k[0],method=k[1],measure=k[2],left_out_subject=s,F=r['F'],p=r['p_unc'],n=r['n']))
o=pd.DataFrame(rows)
o['q']=o.groupby(['left_out_subject','domain']).p.transform(lambda p:multipletests(p,method='fdr_bh')[1])
m=L.merge(o,on=['domain','method','measure','left_out_subject'],suffixes=('','_re'))
assert len(m)==len(L)==5166
res=dict(rows=len(m),max_rel_F=float((m.F_re/m.F-1).abs().max()),max_abs_logp=float((np.log(m.p_re)-np.log(m.p)).abs().max()),
         max_abs_q=float((m.q-m.q_BH_within_domain).abs().max()),n_match=bool((m.n==m.n_participants).all()))
assert res['max_rel_F']<1e-8 and res['max_abs_q']<1e-10 and res['n_match']
import json;print(json.dumps(res,indent=1));json.dump(res,open(Path(__file__).resolve().parent/'results/ORIGINAL_REPRODUCTION.json','w'),indent=1)
