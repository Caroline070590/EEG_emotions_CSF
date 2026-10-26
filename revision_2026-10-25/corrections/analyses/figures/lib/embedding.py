"""
embedding.py — low-dimensional embedding, representational similarity (RSA),
and the emotion-similarity MULTIGRAPH.

  * umap_embed / pca_embed     trial-level embedding of connectivity vectors
  * emotion_rdm                emotion x emotion representational dissimilarity
  * emotion_multigraph         force-directed graph; emotions = nodes,
                               edges weighted by representational similarity.
                               Supports MULTIPLE representations as a multigraph
                               (one edge layer per representation).
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lib.config import EMOS, EI

def umap_embed(agg_feat, n_neighbors=25, min_dist=0.25, metric="cosine", seed=42):
    import umap
    Xs = StandardScaler().fit_transform(agg_feat)
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     metric=metric, random_state=seed).fit_transform(Xs)

def pca_embed(agg_feat, n_components=2):
    Xs = StandardScaler().fit_transform(agg_feat)
    return PCA(n_components=n_components, random_state=0).fit_transform(Xs)

def emotion_rdm(agg_feat, emo_idx):
    """1 - correlation between per-emotion centroid connectivity vectors."""
    cent = np.vstack([agg_feat[emo_idx == EI[e]].mean(0) for e in EMOS])
    return pd.DataFrame(1 - np.corrcoef(cent), index=EMOS, columns=EMOS)

def build_multigraph(rdm_by_method, keep_top=0.45):
    """
    Emotion-similarity multigraph.
      rdm_by_method : dict {method_name: emotion RDM DataFrame}
    Returns a networkx.MultiGraph: nodes = emotions, one edge layer per method,
    edge weight = similarity (1 - normalised dissimilarity); only the strongest
    `keep_top` fraction of edges per layer are kept (for a readable graph).
    """
    G = nx.MultiGraph()
    G.add_nodes_from(EMOS)
    for method, rdm in rdm_by_method.items():
        D = rdm.values.copy()
        D = (D + D.T) / 2; np.fill_diagonal(D, np.nan)
        finite = D[np.isfinite(D)]
        thr = np.nanquantile(finite, keep_top)        # keep most-similar edges
        for i in range(len(EMOS)):
            for j in range(i + 1, len(EMOS)):
                d = D[i, j]
                if d <= thr:
                    sim = 1 - (d - np.nanmin(finite)) / (np.nanmax(finite) - np.nanmin(finite) + 1e-12)
                    G.add_edge(EMOS[i], EMOS[j], key=method, weight=float(sim), method=method)
    return G

def multigraph_layout(G, seed=1):
    """Shared spring layout using summed edge weights across layers."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if H.has_edge(u, v):
            H[u][v]["weight"] += w
        else:
            H.add_edge(u, v, weight=w)
    return nx.spring_layout(H, weight="weight", seed=seed, k=0.9)


def emotion_pca_map(agg, emo_idx, subjects, n_boot=500, seed=42):
    """2D PCA map of the 9 emotion centroids (the spatial form of the RSA),
    with subject-bootstrap clouds for confidence ellipses.

    Returns: ref (9x2 reference coords), boot (dict emotion-index -> Nx2 array).
    PCA is fit on the full-data centroids; bootstrap centroids are projected on
    those fixed axes (linear projection => stable, no axis-flip)."""
    from sklearn.decomposition import PCA
    import numpy as np
    n = len(EMOS)
    cent = np.vstack([agg[emo_idx == EI[e]].mean(0) for e in EMOS])
    pca = PCA(n_components=2, random_state=0).fit(cent)
    ref = pca.transform(cent)
    subs = np.unique(subjects)
    sub_rows = {s: np.where(subjects == s)[0] for s in subs}
    rng = np.random.default_rng(seed)
    boot = {i: [] for i in range(n)}
    for _ in range(n_boot):
        rows = np.concatenate([sub_rows[s] for s in rng.choice(subs, len(subs), replace=True)])
        ei = emo_idx[rows]
        c = np.vstack([agg[rows][ei == EI[e]].mean(0) if (ei == EI[e]).any()
                       else cent[k] for k, e in enumerate(EMOS)])
        p = pca.transform(c)
        for i in range(n):
            boot[i].append(p[i])
    boot = {i: np.array(v) for i, v in boot.items()}
    var = pca.explained_variance_ratio_
    return ref, boot, var
