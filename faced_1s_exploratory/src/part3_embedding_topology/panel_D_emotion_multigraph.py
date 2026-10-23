"""Panel D — emotion-similarity MULTIGRAPH (force-directed).
Nodes = emotions; one edge LAYER per representation (Spearman/MTF/QG if cached).
Edge thickness ∝ representational similarity. This is the 'embedding multigraph'."""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, networkx as nx
from _panels import standalone
from lib.config import CACHE, EMOS, EMO_COLORS, VAL, VAL_COLORS
from lib.embedding import build_multigraph, multigraph_layout

LAYER_STYLE = {"spearman": dict(color="#34495e", style="solid"),
               "mtf":      dict(color="#c0392b", style="dashed"),
               "qg":       dict(color="#16a085", style="dotted")}

def _available_rdms(default_tag):
    rdms = {}
    for t in ["spearman","mtf","qg"]:
        f = CACHE / f"rdm_{t}.csv"
        if f.exists():
            rdms[t] = pd.read_csv(f, index_col=0).reindex(index=EMOS, columns=EMOS)
    if not rdms:  # fall back to whatever tag we have
        rdms[default_tag] = pd.read_csv(CACHE / f"rdm_{default_tag}.csv", index_col=0)
    return rdms

def draw(ax, D, tag):
    rdms = _available_rdms(tag)
    G = build_multigraph(rdms, keep_top=0.5)
    pos = multigraph_layout(G)
    # edges by layer
    for layer, st in LAYER_STYLE.items():
        ews = [(u,v,d["weight"]) for u,v,d in G.edges(data=True) if d.get("method")==layer]
        if not ews: continue
        nx.draw_networkx_edges(G, pos, ax=ax,
            edgelist=[(u,v) for u,v,_ in ews],
            width=[0.4+2.6*w for *_,w in ews],
            edge_color=st["color"], style=st["style"], alpha=0.55)
    # nodes coloured by emotion, ringed by valence
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=240,
        node_color=[EMO_COLORS[n] for n in G.nodes()],
        edgecolors=[VAL_COLORS[VAL[n]] for n in G.nodes()], linewidths=1.4)
    nx.draw_networkx_labels(G, pos, ax=ax,
        labels={n:n[:3] for n in G.nodes()}, font_size=4.5, font_weight="bold", font_color="white")
    ax.set_title("Emotion-similarity multigraph"); ax.axis("off")
    handles = [ax.plot([],[], color=LAYER_STYLE[t]["color"], ls=LAYER_STYLE[t]["style"],
                       lw=1.3, label=t)[0] for t in rdms]
    ax.legend(handles=handles, fontsize=5, loc="lower left", framealpha=0.85, title="layer", title_fontsize=5)

def main(tag="spearman"): standalone(draw, "panelD_multigraph", tag)
if __name__ == "__main__": main()
