"""Step 1: participant x emotion persistence features (H0/H1) at q = 30, using the unchanged release code.

Spearman: 30-channel rank-correlation networks. MTF/QG: q = 30 quantile-state transition matrices
(lib.config.Q_EMBED = 30). Trial matrices are averaged within participant x emotion (lib.representations.aggregate)
and summarized with lib.topology.tda_features, exactly as in part3_embedding_topology/compute_features.py.
Needs the 1-s export in ../src/EEG-data-raw (see ../README.md). Writes inputs/tda_{spearman,mtf,qg}_q30.csv;
intermediate representations go to ./cache (not version-controlled).
"""
import sys
from pathlib import Path
import pandas as pd
B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parent / "src"))
import lib.config as cfg
assert cfg.Q_EMBED == 30 and cfg.REPR_N == {"spearman": 30, "mtf": 30, "qg": 30}
import lib.representations as rep
rep.CACHE = B / "cache"; rep.CACHE.mkdir(exist_ok=True)
from lib.topology import tda_features

for tag in ["spearman", "mtf", "qg"]:
    feat, meta = rep.build_representation(tag=tag, n_jobs=10)
    assert len(meta) == 3444, len(meta)
    agg, am = rep.aggregate(feat, meta)
    tda = pd.concat([am.reset_index(drop=True), tda_features(agg, cfg.REPR_N[tag], n_jobs=10)], axis=1)
    tda.to_csv(B / "inputs" / f"tda_{tag}_q30.csv", index=False)
    print(tag, feat.shape, tda.shape, flush=True)
