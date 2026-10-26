"""Recompute dependent numerical checks from the verified one-second signals."""
from pathlib import Path
import json, hashlib, sys, subprocess
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'corrected_pipeline/src'))
from lib.stats import boot_vs_neutral
from lib.config import EMOS
from verify_raw_lineage import METRICS,REGIONS
LABELS=dict(zip(METRICS,['PLV','PLI','wPLI','AEC','MaxXC','Lag']))
RELEASE=ROOT/'EEG_emotions_CSF_release/faced_1s_exploratory'
LEGACY=ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def zw(s):return (s-s.mean())/(s.std()+1e-9)

def compare(new,old,keys,cols):
    a=new.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
    assert a.index.equals(b.index),(a.index,b.index)
    return {c:float(np.nanmax(np.abs(a[c].to_numpy()-b[c].to_numpy()))) for c in cols}

def main():
    glob=pd.read_csv(HERE/'raw_recomputed_global.csv')
    regional=pd.read_csv(HERE/'raw_recomputed_regional.csv')
    rows=[]
    for metric in METRICS:
        d=glob[['subject','emotion',metric]].copy()
        means=d.groupby('subject')[metric].transform('mean');sd=d.groupby('subject')[metric].transform('std')
        d['y']=np.where(sd>0,(d[metric]-means)/sd.where(sd>0,1.),0.)
        model=smf.ols("y ~ C(emotion, Treatment('neutral'))",d).fit(cov_type='cluster',cov_kwds={'groups':d.subject})
        for e in EMOS:
            if e=='neutral':continue
            key=f"C(emotion, Treatment('neutral'))[T.{e}]"
            rows.append({'measure':LABELS[metric],'emotion':e,'beta':model.params[key],'p_nominal':model.pvalues[key]})
    contrast=pd.DataFrame(rows);contrast['q_fdr']=multipletests(contrast.p_nominal,method='fdr_bh')[1]
    contrast['fdr_significant']=contrast.q_fdr<.05
    contrast.to_csv(HERE/'synchronization_contrasts_1s_recomputed.csv',index=False)
    old=pd.read_csv(RELEASE/'tables/synchronization_contrasts_1s.csv')
    check=compare(contrast,old,['measure','emotion'],['beta','p_nominal','q_fdr'])
    survivor_set=lambda d:set(zip(d.loc[d.q_fdr<.05,'measure'],d.loc[d.q_fdr<.05,'emotion']))
    assert survivor_set(contrast)==survivor_set(old)

    coupling=[];split=[];panel=[]
    rng=np.random.default_rng(2026);subjects=np.array(sorted(regional.subject.unique()))
    parts=[rng.permutation(subjects) for _ in range(300)]
    def slope(t,s):
        s=s-s.mean();return float((s*(t-t.mean())).sum()/(s*s).sum())
    for region in REGIONS:
        d=regional[regional.region==region].copy()
        d['topo_z']=d.groupby('subject').nodal_strength.transform(zw)
        t=d.topo_z.to_numpy();sid=d.subject.to_numpy()
        for metric in METRICS:
            d['sync_z']=d.groupby('subject')[metric].transform(zw);s=d.sync_z.to_numpy()
            m=smf.ols('topo_z ~ sync_z',d).fit(cov_type='cluster',cov_kwds={'groups':d.subject})
            coupling.append({'region':region,'sync_metric':metric,'pooled_beta':m.params['sync_z'],'pooled_p':m.pvalues['sync_z']})
            full=slope(t,s)
            halves=[];both=[]
            for p in parts:
                mask=np.isin(sid,p[:len(p)//2])
                a,b=slope(t[mask],s[mask]),slope(t[~mask],s[~mask])
                halves.extend([a,b]);both.append(np.sign(a)==np.sign(full) and np.sign(b)==np.sign(full))
            split.append({'region':region,'metric':LABELS[metric],'full_beta':full,'half_beta_sd':np.std(halves,ddof=1),
                          'half_beta_2.5pct':np.percentile(halves,2.5),'half_beta_97.5pct':np.percentile(halves,97.5),
                          'directional_replication_pct':100*np.mean(both),'n_partitions':300})
        strength=d.groupby(['subject','emotion']).nodal_strength.mean().groupby('emotion').mean()
        beta=[c['pooled_beta'] for c in coupling if c['region']==region and c['sync_metric']=='wpli'][0]
        pvalue=[c['pooled_p'] for c in coupling if c['region']==region and c['sync_metric']=='wpli'][0]
        panel.append({'region':region,'nodal_strength_range':round(strength.max()-strength.min(),4),
                      'mean_wpli':round(d.wpli.mean(),4),'coupling_beta':round(beta,4),'coupling_p':pvalue})
    coupling=pd.DataFrame(coupling);coupling.to_csv(HERE/'regional_coupling_recomputed.csv',index=False)
    split=pd.DataFrame(split);split.to_csv(HERE/'regional_coupling_splithalf_recomputed.csv',index=False)
    panel=pd.DataFrame(panel);panel.to_csv(HERE/'figure5g_values_recomputed.csv',index=False)
    oldcoupling=pd.read_csv(ROOT/'emotion-eeg-networks/_staging/q30_final_analysis/csv/TableS_region_synchrony_topology_coupling.csv')
    oldsplit=pd.read_csv(RELEASE/'tables/regional_coupling_splithalf.csv')
    oldpanel=pd.read_csv(ROOT/'emotion-eeg-networks/_staging/q30_final_analysis/csv/Table_fig4_panel_f_combined.csv')
    region_check=compare(coupling,oldcoupling,['region','sync_metric'],['pooled_beta','pooled_p'])
    split_check=compare(split,oldsplit,['region','metric'],['full_beta','half_beta_sd','half_beta_2.5pct','half_beta_97.5pct','directional_replication_pct'])
    panel_check=compare(panel,oldpanel,['region'],['nodal_strength_range','mean_wpli','coupling_beta','coupling_p'])

    boots=[]
    for e in EMOS:
        if e=='neutral':continue
        b,lo,hi=boot_vs_neutral(glob,'wpli',e,n_boot=2000)
        boots.append({'emotion':e,'d':b,'ci_lo':lo,'ci_hi':hi})
    boots=pd.DataFrame(boots);boots.to_csv(HERE/'wpli_bootstrap_vs_neutral_recomputed.csv',index=False)
    oldboots=pd.read_csv(LEGACY/'cache/bootstrap_effects_vs_neutral.csv')
    oldboots=oldboots[(oldboots.domain=='synchrony')&(oldboots.measure=='wpli')]
    boots_check=compare(boots,oldboots,['emotion'],['d','ci_lo','ci_hi'])
    pairs=[]
    for i,e in enumerate(EMOS):
        for j,f in enumerate(EMOS):
            if i<=j:continue
            b,lo,hi=boot_vs_neutral(glob,'wpli',e,ref=f,n_boot=800)
            pairs.append({'emotion_a':e,'emotion_b':f,'d':b,'ci_lo':lo,'ci_hi':hi})
    pairs=pd.DataFrame(pairs);pairs.to_csv(HERE/'wpli_pairwise_recomputed.csv',index=False)
    oldpairs=pd.read_csv(LEGACY/'cache/pairwise_emotion_comparisons.csv')
    oldpairs=oldpairs[(oldpairs.domain=='synchrony')&(oldpairs.measure=='wpli')]
    pairs_check=compare(pairs,oldpairs,['emotion_a','emotion_b'],['d','ci_lo','ci_hi'])

    result={'status':'dependent numerical comparisons complete','global48':check,
            'global48_fdr_discoveries':len(survivor_set(contrast)),'global48_fdr_discovery_set_unchanged':True,
            'regional30_pooled':region_check,'regional30_split_half':split_check,'figure5g':panel_check,
            'wpli8_bootstrap_vs_neutral':boots_check,'wpli36_pairwise':pairs_check,
            'script_sha256':sha(__file__),
            'limitations':['Global coupling Figure4 panels b-g still require independent generator/output linkage; this table certifies regional coupling, not every global MixedLM coefficient.',
                           'The historical LOPO generator remains unrecovered; no full rerun is claimed.']}
    (HERE/'DOWNSTREAM_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
