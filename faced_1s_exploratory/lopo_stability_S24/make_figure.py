"""Step 3: Supplementary Fig. S24 from results/lopo_summary_all42.csv and results/lopo_long.csv (no typed values)."""
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;O=B/'results'
s=pd.read_csv(O/'lopo_summary_all42.csv');s=s[s.displayed].reset_index(drop=True)
L=pd.read_csv(O/'lopo_long.csv');key=['domain','representation','measure']
COL={'topology':'#0072B2','synchronization':'#D55E00','higher-order topology':'#CC79A7'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Nimbus Sans','DejaVu Sans'],'font.size':8,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(7.2,0.33*len(s)+1.2));rng=np.random.default_rng(0)
for i,r in s.iterrows():
    y=len(s)-1-i;g=L.merge(pd.DataFrame([r[key]]),on=key);v=-np.log10(g.p_gg.values)
    ax.scatter(v,y+rng.uniform(-.18,.18,len(v)),s=6,color=COL[r.domain],alpha=.35,lw=0,zorder=2)
    ax.scatter(-np.log10(r.full_p_gg),y,s=34,marker='D',color=COL[r.domain],edgecolor='k',lw=.6,zorder=3)
    ax.text(1.005,y,f"{r.lopo_pct_q_gg_lt05:.0f}%",transform=ax.get_yaxis_transform(),va='center',fontsize=7)
ax.axvline(-np.log10(.05),color='0.4',lw=.8,ls='--',zorder=1)
ax.text(-np.log10(.05),len(s)-.35,' p = 0.05',fontsize=6.5,color='0.3',va='bottom')
ax.text(1.005,len(s)-.35,'q < 0.05',transform=ax.get_yaxis_transform(),fontsize=6.5,color='0.3',va='bottom')
ax.set_yticks(range(len(s)));ax.set_yticklabels(s.label[::-1]);ax.set_ylim(-.7,len(s)-.1)
ax.set_xlim(0,max(14,np.ceil(-np.log10(L.p_gg.min()))+.5))
ax.set_xlabel(r'Omnibus emotion effect, $-\log_{10}(p_{\mathrm{GG}})$');ax.grid(axis='x',color='0.92',lw=.6,zorder=0)
fig.tight_layout();fig.subplots_adjust(right=0.9)
for ext,kw in [('pdf',{}),('svg',{}),('png',{'dpi':600})]:fig.savefig(B/'figures'/f'FigS24_LOPO_stability_q30.{ext}',**kw)
print(len(s),'rows')
