"""Step 2: LOPO stability of omnibus emotion effects on current (q = 30, corrected-synchronization) inputs. See PROTOCOL.md."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from lopo_lib import rm_anova
I=B/'inputs';O=B/'results'
ROOT=B.parents[1]
TOPO=ROOT/'revision_2026-10-25/provenance/critical_review/figure1/src/cache'   # q = 30 tables behind main Fig. 1
SYNC_CSV=ROOT/'faced_1s_exploratory/derived/trial_scalar_synchronization.csv'
METRICS=['sigma_small_world','gamma_C_over_Crand','lambda_L_over_Lrand','transitivity','char_path_len_gcc','global_efficiency']
TCOLS=['H0_total','H0_entropy','H1_n','H1_total','H1_max','H1_entropy']
SYNC={'PLV':'PLV','PLI':'PLI','wPLI':'wPLI','AEC':'AEC','MaxXC':'MaxXC','Lag':'Lag'}
REP={'spearman':'Spearman','mtf':'MTF','qg':'QG'}
LAB={'sigma_small_world':'σ','gamma_C_over_Crand':'γ','lambda_L_over_Lrand':'λ','transitivity':'transitivity','char_path_len_gcc':'path length',
     'global_efficiency':'efficiency','H0_total':'H0 total','H0_entropy':'H0 entropy','H1_n':'H1 count','H1_total':'H1 total','H1_max':'H1 max','H1_entropy':'H1 entropy'}

def cells():
    out={}
    for rep in REP:
        d=pd.read_csv(TOPO/f'trial_metrics_{rep}.csv').replace([np.inf,-np.inf],np.nan)
        assert len(d)==3444 and set(d.q)=={30} and set(d.density)=={0.1}
        for m in METRICS:out[('topology',rep,m)]=d.groupby(['subject','emotion'])[m].mean().unstack()
    s=pd.read_csv(SYNC_CSV);assert len(s)==3444
    s['subject']=s.subject_id+1
    for m in SYNC:out[('synchronization','all pairs',m)]=s.groupby(['subject','emotion'])[m].mean().unstack()
    for rep in REP:
        t=pd.read_csv(I/f'tda_{rep}_q30.csv');assert len(t)==1107
        t['subject']=t.subject.str.replace('Patient','').astype(int)
        for m in TCOLS:out[('higher-order topology',rep,m)]=t.set_index(['subject','emotion'])[m].unstack()
    for c in out.values():assert c.shape==(123,9)
    return out

def run(cs,drop=None):
    rows=[]
    for (dom,rep,m),c in cs.items():
        r=rm_anova(c if drop is None else c.drop(index=drop));rows.append(dict(domain=dom,representation=rep,measure=m,left_out=drop,**r))
    o=pd.DataFrame(rows)
    for p in ['p_gg','p_unc']:o['q_'+p[2:]]=o.groupby('domain')[p].transform(lambda x:multipletests(x,method='fdr_bh')[1])
    return o

def main():
    cs=cells();subjects=list(next(iter(cs.values())).index)
    full=run(cs);lopo=pd.concat([run(cs,s) for s in subjects],ignore_index=True)
    assert len(lopo)==123*42
    full.to_csv(O/'full_sample_omnibus.csv',index=False);lopo.to_csv(O/'lopo_long.csv',index=False)
    key=['domain','representation','measure']
    g=lopo.assign(nlp=-np.log10(lopo.p_gg)).groupby(key,sort=False)
    summ=full[key+['n','F','eps_gg','ges','p_gg','q_gg','p_unc','q_unc']].rename(columns=lambda c:c if c in key else 'full_'+c).merge(
        g.agg(lopo_runs=('p_gg','size'),lopo_n_min=('n','min'),lopo_n_max=('n','max'),lopo_max_p_gg=('p_gg','max'),
              lopo_median_nlp=('nlp','median'),lopo_min_nlp=('nlp','min'),lopo_max_nlp=('nlp','max'),
              lopo_pct_q_gg_lt05=('q_gg',lambda x:100*(x<.05).mean()),lopo_pct_q_unc_lt05=('q_unc',lambda x:100*(x<.05).mean()),
              lopo_ges_min=('ges','min'),lopo_ges_max=('ges','max')).reset_index(),on=key)
    summ['label']=[f"{'Synchrony' if d=='synchronization' else ('Higher-order' if d.startswith('higher') else 'Topology')}: {'all pairs '+m if d=='synchronization' else REP_L(r)+' '+LAB[m]}" for d,r,m in zip(summ.domain,summ.representation,summ.measure)]
    summ['displayed']=summ.full_q_gg<.05
    summ.to_csv(O/'lopo_summary_all42.csv',index=False)
    print(summ[['label','full_n','full_F','full_eps_gg','full_q_gg','full_q_unc','lopo_min_nlp','lopo_pct_q_gg_lt05','lopo_pct_q_unc_lt05','displayed']].round(4).to_string(index=False))
    json.dump(dict(n_measures=42,n_lopo=len(subjects),n_displayed=int(summ.displayed.sum()),
                   displayed_all_100pct=bool((summ[summ.displayed].lopo_pct_q_gg_lt05==100).all())),open(O/'SUMMARY.json','w'),indent=1)

REP_L=lambda r:{'spearman':'Spearman','mtf':'MTF','qg':'QG'}.get(r,r)
if __name__=='__main__':main()
