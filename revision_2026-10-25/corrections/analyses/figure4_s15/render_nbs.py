"""Re-render retained 1-s NBS networks with nonzero p/q labels; no permutations."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
CH=['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4',
    'T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']
EMOS=['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def probability_label(symbol,value):
    if not 0 < value <= 1:raise ValueError("Expected positive probability")
    return f"{symbol}<0.001" if value < .001 else f"{symbol}={value:.3f}"


def main():
    spec=importlib.util.spec_from_file_location("original_scalp_layout",HERE/"provenance/layout_original.py")
    layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
    agg=np.load(HERE/"inputs/agg_spearman.npy")
    meta=pd.read_csv(HERE/"inputs/agg_meta_spearman.csv")
    tab=pd.read_csv(HERE/"inputs/nbs_spearman_vs_neutral.csv").set_index("emotion")
    assert agg.shape==(1107,435) and len(meta)==1107
    assert set(meta.subject)=={f"Patient{i:03d}" for i in range(1,124)}
    assert (pd.crosstab(meta.subject,meta.emotion).to_numpy()==1).all()
    np.testing.assert_allclose(tab.q_bh_across_emotions,
        multipletests(tab.p_cluster_fwe,method="fdr_bh")[1],atol=1e-15)
    assert (tab.t_threshold==3).all() and (tab.n_permutations==5000).all()
    assert np.allclose(tab.p_cluster_fwe*5001,np.round(tab.p_cluster_fwe*5001))
    originals=HERE.parents[1]/"manuscript/Figures/Figure_D1_NBSSubnetworks.pdf"
    original_text=subprocess.run(["pdftotext",str(originals),"-"],capture_output=True,text=True,check=True).stdout
    iu=np.triu_indices(30,1);records=[];edge_rows=[]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(3,3,figsize=(7.2,7.5))
    fig.subplots_adjust(left=.025,right=.975,top=.94,bottom=.085,hspace=.28,wspace=.09)
    for ax,emo in zip(axes.ravel(),EMOS):
        if emo=="neutral":
            layout.head(ax);ax.set_title("Neutral\n(reference)",fontsize=11,pad=6)
            continue
        row=tab.loc[emo]
        differences=[]
        for s in meta.subject.unique():
            a=(meta.subject==s)&(meta.emotion==emo);b=(meta.subject==s)&(meta.emotion=="neutral")
            differences.append(agg[a.to_numpy()][0]-agg[b.to_numpy()][0])
        d=np.vstack(differences)
        t=d.mean(0)/((d.std(0,ddof=1)+1e-12)/np.sqrt(len(d)))
        supra=abs(t)>row.t_threshold;g=nx.Graph()
        for idx in np.flatnonzero(supra):g.add_edge(int(iu[0][idx]),int(iu[1][idx]),index=int(idx))
        component=max(nx.connected_components(g),key=lambda c:g.subgraph(c).number_of_edges())
        edges=list(g.subgraph(component).edges());signs=[]
        assert len(edges)==int(row.component_edges)
        assert f"{emo} ({len(edges)} edges" in original_text.replace("  "," ")
        for i,j in edges:
            idx=g[i][j]["index"];sign=float(np.sign(t[idx]));signs.append(sign)
            edge_rows.append(dict(emotion=emo,channel_i=CH[i],channel_j=CH[j],t=float(t[idx]),sign=int(sign)))
        layout.scalp_edges(ax,CH,edges,signs)
        p=probability_label("p",row.p_cluster_fwe);q=probability_label("q",row.q_bh_across_emotions)
        star="*" if row.q_bh_across_emotions<.05 else ""
        ax.set_title(f"{emo.capitalize()}: {len(edges)} edges{star}\n{p}; {q}",fontsize=10,pad=6)
        records.append(dict(emotion=emo,n_edges=len(edges),p=float(row.p_cluster_fwe),q=float(row.q_bh_across_emotions),
                            displayed_p=p,displayed_q=q))
    handles=[Line2D([0],[0],color="#c0392b",lw=1.6,label="Larger absolute Spearman dependence"),
             Line2D([0],[0],color="#2c5fa8",lw=1.6,label="Smaller absolute Spearman dependence")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.016),frameon=False,fontsize=9,ncol=1)
    out=HERE/"outputs";out.mkdir(exist_ok=True)
    pdf=out/"Figure_D1_NBSSubnetworks_readable.pdf";png=pdf.with_suffix(".png")
    fig.savefig(pdf,bbox_inches="tight");fig.savefig(png,dpi=220,bbox_inches="tight");plt.close(fig)
    text=subprocess.run(["pdftotext",str(pdf),"-"],capture_output=True,text=True,check=True).stdout
    assert "p=0.000" not in text and "q=0.000" not in text
    assert text.count("p<0.001")==text.count("q<0.001")==4
    pd.DataFrame(records).to_csv(out/"NBS_displayed_statistics.csv",index=False)
    pd.DataFrame(edge_rows).to_csv(out/"NBS_plotted_edges.csv",index=False)
    audit={"status":"passed","scope":"Same retained 1-s Spearman NBS results; display and typography only. No permutations or inferential fits were rerun.",
        "source_sha256":sha(Path(__file__)),"original_pdf_sha256":sha(originals),
        "input_sha256":{p.name:sha(p) for p in sorted((HERE/"inputs").iterdir()) if p.is_file()},
        "original_layout_sha256":sha(HERE/"provenance/layout_original.py"),
        "all_eight_component_edge_counts_match_original_PDF_and_cached_statistics":True,
        "all_plotted_edges_reconstructed_from_saved_participant_condition_edge_averages":True,
        "saved_BH_q_values_independently_recalculated":True,
        "original_probability_values_unchanged":True,"small_pq_label_counts":{"p<0.001":4,"q<0.001":4},
        "significance_marker":"Asterisk denotes the saved BH q<0.05 across eight condition comparisons; p is component-level permutation FWE p.",
        "pdf_sha256":sha(pdf),"png_sha256":sha(png),"values":records}
    (out/"NBS_VALIDATION.json").write_text(json.dumps(audit,indent=2)+"\n")
    (out/"NBS_caption_replacement.txt").write_text("The number of edges, component-level permutation FWE p-value, and BH-adjusted q-value across the eight condition comparisons are shown above each topography. Values below 0.001 are displayed as p<0.001 or q<0.001; asterisks denote q<0.05. Colors indicate larger (red) or smaller (blue) absolute Spearman dependence relative to neutral, rather than the sign of the underlying correlation.\n")
    print(json.dumps({"status":"passed","output":str(pdf),"unchanged_statistics":records},indent=2))


if __name__=="__main__":main()
