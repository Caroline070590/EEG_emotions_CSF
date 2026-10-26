"""
_panels.py — shared helpers for Part 3 panel scripts.

Each panel module exposes:
    draw(ax, D, tag)     draw the panel into a given Axes (used by run_part3)
    main()               render the panel STANDALONE into figures/part3/

`load(tag)` returns a dict D with every cached artefact a panel might need.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, FIG_DIR, set_nature_style, save, W1, EMOS, EI

def load(tag="spearman"):
    D = dict(
        agg = np.load(CACHE / f"agg_{tag}.npy"),
        am  = pd.read_csv(CACHE / f"agg_meta_{tag}.csv"),
        tda = pd.read_csv(CACHE / f"tda_{tag}.csv"),
        rdm = pd.read_csv(CACHE / f"rdm_{tag}.csv", index_col=0).reindex(index=EMOS, columns=EMOS),
        lmm = pd.read_csv(CACHE / f"lmm_{tag}.csv"),
        boot = json.load(open(CACHE / f"boot_{tag}.json")),
        tag = tag,
    )
    D["emo_idx"] = D["am"]["emotion"].map(EI).values
    return D

def standalone(draw_fn, name, tag="spearman", figsize=(W1, W1)):
    import matplotlib.pyplot as plt
    set_nature_style()
    D = load(tag)
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax, D, tag)
    save(fig, FIG_DIR / "part3" / f"{name}_{tag}")
    plt.close(fig)

def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
