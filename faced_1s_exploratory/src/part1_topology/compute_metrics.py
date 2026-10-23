"""
compute_metrics.py — Part 1 driver. Computes Spearman/MTF/QG small-world metrics
for every trial and writes cache/trial_metrics_{method}.csv.

Checkpointed & resumable: results are appended in batches, completed trials are
skipped on re-run, so it can be stopped and restarted (or run in the background).

    python part1_topology/compute_metrics.py --n_jobs -1 --n_rand 20
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joblib import Parallel, delayed

from lib.config import CACHE
from lib.metrics import _metrics_one
from lib.representations import _list_trials

FILES = {"Spearman": CACHE/"trial_metrics_spearman.csv",
         "MTF": CACHE/"trial_metrics_mtf.csv", "QG": CACHE/"trial_metrics_qg.csv"}

def _done_keys():
    done = set()
    for f in FILES.values():
        if f.exists():
            d = pd.read_csv(f, usecols=["subject", "trial"])
            done |= set(zip(d["subject"], d["trial"]))
    return done

def run(n_jobs=-1, n_rand=20, batch=120):
    items = _list_trials()
    done = _done_keys()
    todo = [(p, emo) for p, subj, emo in items
            if (int(__import__("re").search(r"Patient[_\-]?(\d+)", p.parent.name, 2).group(1)),
                p.stem) not in done]
    print(f"total {len(items)} | done {len(done)} | todo {len(todo)}", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i+batch]
        out = Parallel(n_jobs=n_jobs)(delayed(_metrics_one)(p, emo, n_rand) for p, emo in chunk)
        for key, f in FILES.items():
            df = pd.DataFrame([r[key] for r in out])
            df.to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"  +{len(chunk)}  ({i+len(chunk)}/{len(todo)})", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_rand", type=int, default=20)
    a = ap.parse_args()
    run(n_jobs=a.n_jobs, n_rand=a.n_rand)
