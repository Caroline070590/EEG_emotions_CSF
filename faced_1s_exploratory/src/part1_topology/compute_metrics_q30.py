"""
metrics_q30.py
--------------------------------------------------------------------------
Addresses the following caveat:

    "Spearman networks have 30 nodes; MTF/QG state-transition graphs have
    only 13 (the quantile states), and at 10% density that's ~8 edges on
    a 78-edge-max graph. Degree-preserving double-edge swaps on graphs
    that small/sparse are a much less stable null model than on a
    30-node graph -- gamma, lambda, and sigma for MTF/QG are going to be
    noisier by construction, independent of any real emotion effect."

The original q for MTF/QG (q=13 for T=250 samples) comes from the
data-driven Campanharo formula q = round(2*T^(1/3)), clipped to [3,30]
(lib.representations.choose_q_campanharo). That is a principled choice
for the SYMBOLIC DYNAMICS itself (how finely to discretize the amplitude
distribution), but it was never chosen to match Spearman's 30-channel
node count, and the mismatch is exactly what makes the small-world
randomization noisier for MTF/QG.

This module reruns the SAME metrics pipeline as lib.metrics, with q fixed
at 30 (the upper clip bound already used by choose_q_campanharo, so nothing
here is "invented" -- 30 is already a value the formula is allowed to
produce for longer trials) instead of the data-driven 13. Node count then
matches Spearman's 30 exactly, target edges at 10% density go from ~8 to
~44 (matching Spearman's graph size almost exactly), and the null-model
randomization has as much material to work with as the Spearman graphs do.

The trade-off, reported honestly rather than hidden: with only T=250
samples per trial, q=30 quantile states means ~8.3 samples per state per
channel (vs ~19 at q=13), so the underlying quantile-state discretization
itself is coarser-grained per bin. This script lets you compare q=13 vs
q=30 directly (fingerprint reliability, small-world values) so you can see
whether that trade-off nets out favourably.

Usage
=====
    python3 part1_topology/compute_metrics_q30.py --n_jobs 1 --n_rand 20

Writes cache/trial_metrics_mtf_q30.csv and cache/trial_metrics_qg_q30.csv,
using the exact same threshold_to_density / _rand_refs / metrics_from_matrix
functions as the original q=13 pipeline (lib.metrics), so results are
directly comparable cell-for-cell.
"""
import sys, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, N_RANDOMIZATIONS
from lib.metrics import metrics_from_matrix
from lib.representations import (load_raw_trial_csv, parse_subject_trial,
                                  choose_k_from_autocorrelation,
                                  matrix_markov_transition, _list_trials)

Q_FIXED = 30  # matches Spearman's 30-channel node count; also the upper
              # clip bound already used by choose_q_campanharo elsewhere.


def _metrics_one_q30(path, emo, n_rand):
    X = load_raw_trial_csv(path)
    subj, tr = parse_subject_trial(path)
    T = X.shape[0]
    k = choose_k_from_autocorrelation(X)  # QG lag: unchanged, still data-driven
    base = dict(subject=subj, emotion=emo, trial=Path(path).stem, T=T, q=Q_FIXED, k=k)
    res = {}
    res["MTF"] = {**base, **metrics_from_matrix(matrix_markov_transition(X, q=Q_FIXED, lag=1), n_rand=n_rand)}
    res["QG"] = {**base, **metrics_from_matrix(matrix_markov_transition(X, q=Q_FIXED, lag=k), n_rand=n_rand)}
    return res


def _done_keys(files):
    done = set()
    for f in files.values():
        if f.exists():
            d = pd.read_csv(f, usecols=["subject", "trial"])
            done |= set(zip(d["subject"], d["trial"]))
    return done


def compute_trial_metrics_q30(n_jobs=1, n_rand=N_RANDOMIZATIONS, force=False, limit=None):
    """Resumable: writes in append mode and skips (subject, trial) pairs
    already present in the CSVs, so this can safely be invoked repeatedly
    (e.g. in bounded time-limited chunks) until complete. `limit` caps how
    many NEW trials are processed in this call."""
    files = {"mtf_q30": CACHE / "trial_metrics_mtf_q30.csv",
             "qg_q30": CACHE / "trial_metrics_qg_q30.csv"}
    if force:
        for f in files.values():
            if f.exists():
                f.unlink()
    items = _list_trials()
    done = _done_keys(files)
    todo = [(p, subj, emo) for p, subj, emo in items
            if (parse_subject_trial(p)[0], Path(p).stem) not in done]
    print(f"[metrics_q30] total {len(items)} | done {len(done)} | todo {len(todo)}", flush=True)
    if limit is not None:
        todo = todo[:limit]
    t0 = time.time()
    batch = 100
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        rows_mtf, rows_qg = [], []
        for p, subj, emo in chunk:
            r = _metrics_one_q30(p, emo, n_rand)
            rows_mtf.append(r["MTF"]); rows_qg.append(r["QG"])
        pd.DataFrame(rows_mtf).to_csv(files["mtf_q30"], mode="a", header=not files["mtf_q30"].exists(), index=False)
        pd.DataFrame(rows_qg).to_csv(files["qg_q30"], mode="a", header=not files["qg_q30"].exists(), index=False)
        print(f"  +{len(chunk)}  ({i+len(chunk)}/{len(todo)} this call, {time.time()-t0:.0f}s)", flush=True)
    print(f"[metrics_q30] this call processed {len(todo)} trials in {time.time()-t0:.1f}s", flush=True)
    return {m: (pd.read_csv(f) if f.exists() else None) for m, f in files.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--n_rand", type=int, default=N_RANDOMIZATIONS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    compute_trial_metrics_q30(n_jobs=a.n_jobs, n_rand=a.n_rand, force=a.force)
