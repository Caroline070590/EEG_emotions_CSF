"""Readable two-page rendering of verified existing one-second synchrony results.

No inferential fits are run here. Coefficients were recovered from the
historical trial-level participant-clustered OLS pipeline and independently
matched to every cell of the original PDF at its displayed color resolution.
"""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EMOS=['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']
SYNC=['plv','pli','wpli','amp_env_corr','max_cross_corr','lag_ms']
LABELS=['PLV','PLI','wPLI','AEC','MaxXC','Lag']
TOPO=['sigma_small_world','transitivity','char_path_len_gcc','global_efficiency']
TLABELS=['σ','Transitivity','Path length','Efficiency']
COLORS={'anger':'#c0392b','disgust':'#7d3c98','fear':'#34495e','sadness':'#2471a3',
        'neutral':'#7f8c8d','amusement':'#e67e22','inspiration':'#16a085','joy':'#f1c40f','tenderness':'#e84393'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stars(p):return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''


def heatmap(ax,matrix,rows,columns,cmap,limits):
    im=ax.imshow(matrix,cmap=cmap,vmin=limits[0],vmax=limits[1],aspect='auto',interpolation='nearest')
    ax.set_xticks(range(len(columns)));ax.set_xticklabels(columns,fontsize=10)
    ax.set_yticks(range(len(rows)));ax.set_yticklabels([s.capitalize() if s in EMOS else s for s in rows],fontsize=10)
    ax.set_xticks(np.arange(-.5,len(columns),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='both',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    return im


def annotate(ax,pvalues,values,light_threshold=None):
    for i in range(pvalues.shape[0]):
        for j in range(pvalues.shape[1]):
            marker=stars(pvalues[i,j])
            if marker:
                white=values[i,j]<.5 if light_threshold is None else abs(values[i,j])>light_threshold
                ax.text(j,i,marker,ha='center',va='center',fontsize=11,color='white' if white else '#111111')


def panel(ax,letter):ax.text(-.11,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=14,va='bottom')


def colorbar(fig,im,ax,label):
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.ax.tick_params(labelsize=9);cb.set_label(label,fontsize=10)


def save(fig,out,stem):
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',pad_inches=.12)
    fig.savefig(out/(stem+'.png'),dpi=220,bbox_inches='tight',pad_inches=.12);plt.close(fig)


def main():
    inp=HERE/'inputs';out=HERE/'outputs';out.mkdir(exist_ok=True)
    lineage_path=HERE/'provenance/HISTORICAL_GLOBAL_FIGURE_VALIDATION.json'
    lineage=json.loads(lineage_path.read_text())
    assert lineage['n_trial_rows']==3444
    for check in lineage['rendered_checks'].values():assert check['exact_matching_cells']==check['n_cells'] and check['max_RGB_difference']==0
    assert sha(inp/'trial_synchrony_metrics.csv')==lineage['input_sha256']['emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed/cache/trial_synchrony_metrics.csv']
    pooled=pd.read_csv(inp/'coupling_pooled.csv');slopes=pd.read_csv(inp/'coupling_condition_slopes.csv')
    interactions=pd.read_csv(inp/'coupling_interactions.csv');summary=pd.read_csv(inp/'coupling_magnitude.csv')
    sync=pd.read_csv(inp/'trial_synchrony_metrics.csv');contrasts=pd.read_csv(inp/'synchronization_contrasts.csv')
    assert len(sync)==3444 and len(pooled)==24 and len(slopes)==216 and len(interactions)==24
    assert np.isfinite(pooled[['beta','p_nominal']]).all().all() and np.isfinite(slopes[['beta','p_nominal']]).all().all()
    # Exact same two-stage averaging and sample SD used in the old fingerprint.
    means=sync.groupby(['subject','emotion'])[SYNC].mean().groupby('emotion').mean().reindex(EMOS)
    z=((means-means.mean(0))/means.std(0,ddof=1)).to_numpy()
    pa=np.full((9,6),np.nan)
    for j,label in enumerate(LABELS):
        for i,emo in enumerate(EMOS):
            if emo!='neutral':pa[i,j]=contrasts[(contrasts.measure==label)&(contrasts.emotion==emo)].p_nominal.iloc[0]
    B=pooled.pivot(index='topology',columns='synchrony',values='beta').loc[TOPO,SYNC].to_numpy()
    P=pooled.pivot(index='topology',columns='synchrony',values='p_nominal').loc[TOPO,SYNC].to_numpy()
    be=np.stack([slopes[slopes.topology==t].pivot(index='emotion',columns='synchrony',values='beta').loc[EMOS,SYNC].to_numpy() for t in TOPO])
    pe=np.stack([slopes[slopes.topology==t].pivot(index='emotion',columns='synchrony',values='p_nominal').loc[EMOS,SYNC].to_numpy() for t in TOPO])
    ip=interactions.pivot(index='topology',columns='synchrony',values='interaction_p_nominal').loc[TOPO,SYNC].to_numpy()
    magnitude=np.abs(be).mean(axis=(0,2));counts=(pe<.05).sum(axis=(0,2))
    check=summary.set_index('emotion').loc[EMOS]
    np.testing.assert_allclose(magnitude,check.mean_abs_beta,atol=1e-14)
    np.testing.assert_array_equal(counts,check.n_nominal_sig)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.6,6.6));grid=fig.add_gridspec(2,2,height_ratios=[1,1.02],hspace=.60,wspace=.65,
                                                      left=.14,right=.95,top=.93,bottom=.09)
    ax=fig.add_subplot(grid[0,:]);im=heatmap(ax,z,EMOS,LABELS,'magma',(-2,2));annotate(ax,pa,z)
    ax.set_title('Broadband coordination fingerprint',pad=10);panel(ax,'a');colorbar(fig,im,ax,'Column z-score')
    ax=fig.add_subplot(grid[1,0]);im=heatmap(ax,B,TLABELS,LABELS,'PuOr_r',(-.3,.3));annotate(ax,P,B,.18)
    ax.set_title('Pooled coupling',pad=12);panel(ax,'b');colorbar(fig,im,ax,'Standardized β')
    ax.tick_params(axis='x',labelrotation=45)
    ax=fig.add_subplot(grid[1,1]);order=np.argsort(magnitude)[::-1]
    ax.barh(range(9),magnitude[order],color=[COLORS[EMOS[i]] for i in order],edgecolor='black',linewidth=.3,height=.72)
    for k,i in enumerate(order):ax.text(magnitude[i]+.004,k,str(counts[i]),va='center',fontsize=10)
    ax.set_yticks(range(9));ax.set_yticklabels([EMOS[i].capitalize() for i in order]);ax.invert_yaxis()
    ax.set_xlim(0,magnitude.max()*1.25);ax.set_xlabel('Mean |β| across 24 pairs')
    ax.set_title('Coupling magnitude by condition',pad=12,fontsize=10);panel(ax,'c')
    save(fig,out,'Figure_Part2_synchrony_overview_readable')
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.7));fig.subplots_adjust(left=.15,right=.88,top=.93,bottom=.10,wspace=.60,hspace=.47)
    for k,(ax,t) in enumerate(zip(axes.ravel(),TOPO)):
        columns=[label+(' †' if ip[k,j]<.05 else '') for j,label in enumerate(LABELS)]
        im=heatmap(ax,be[k],EMOS,columns,'RdBu_r',(-.2,.2));annotate(ax,pe[k],be[k],.12)
        ax.tick_params(axis='x',labelrotation=45);ax.set_title(TLABELS[k],pad=10);panel(ax,'defg'[k])
    cax=fig.add_axes([.91,.19,.018,.64]);cb=fig.colorbar(im,cax=cax);cb.set_label('Condition-specific standardized β',fontsize=10);cb.ax.tick_params(labelsize=9)
    save(fig,out,'Figure_Part2_synchrony_condition_slopes_readable')
    np.savez_compressed(out/'synchrony_plotted_values.npz',fingerprint=z,fingerprint_p_nominal=pa,
        pooled_beta=B,pooled_p_nominal=P,condition_beta=be,condition_p_nominal=pe,
        interaction_p_nominal=ip,coupling_magnitude=magnitude,nominal_significant_count=counts,
        emotion=np.array(EMOS),synchrony=np.array(SYNC),topology=np.array(TOPO))
    audit={'status':'passed','scope':'Re-rendering of verified original one-second results only; no inferential fits run by this rendering script.',
        'source_sha256':sha(Path(__file__)),'lineage_validation_sha256':sha(lineage_path),
        'lineage_scope':'Historical trial-level participant-clustered OLS reproduces all 294 original heatmap cells at displayed color resolution. Exact hidden floating-point values cannot be reconstructed from the PDF alone; coefficients are reproduced by the identified historical code and verified input caches.',
        'model':'Pooled: topology_z~synchrony_z, intercept included. Condition-specific: topology_z~synchrony_z*C(emotion), participant-clustered OLS. Interaction: 8-df Wald test. Within-participant sample-SD standardization. 3444 trials.',
        'annotations':'All stars, counts, and daggers are nominal, unadjusted comparisons; stars p<.05/.01/.001; daggers interaction p<.05. No FDR interpretation is introduced.',
        'n_fingerprint_cells':54,'n_pooled_coefficients':24,'n_condition_coefficients':216,
        'original_colormaps_and_limits_preserved':True,'coupling_magnitudes_and_counts_match_verified_table':True,
        'input_sha256':{p.name:sha(p) for p in sorted(inp.iterdir()) if p.is_file()},
        'output_sha256':{p.name:sha(p) for p in sorted(out.glob('*synchrony*')) if p.is_file()}}
    (out/'SYNCHRONY_FIGURE_VALIDATION.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'Synchrony_layout_integration.txt').write_text('''Replace the original dense composite with two NEW assets, preserving its figure number via ContinuedFloat:
  Figure_Part2_synchrony_overview_readable.pdf — panels a–c
  Figure_Part2_synchrony_condition_slopes_readable.pdf — panels d–g

The original PDF is unchanged. Both new figures preserve all heatmap values,
color limits, nominal stars, and the summary counts. Panel c numbers are the
number of nominal p<0.05 slopes among 24 pairs. This is not a corrected count.
Daggers mark nominal p<0.05 in 8-df emotion-by-synchronization Wald tests.
The model is trial-level participant-clustered OLS, not MixedLM or an LRT.
Use the main figure caption for all tests; the continued caption may simply
identify panels d–g as condition-specific slopes with nominal stars/daggers.
''')
    print(json.dumps({'status':'passed','n_heatmap_cells':294,'outputs':list(audit['output_sha256'])},indent=2))


if __name__=='__main__':main()
