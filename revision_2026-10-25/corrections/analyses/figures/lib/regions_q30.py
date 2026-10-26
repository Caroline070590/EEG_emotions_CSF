"""
regions_q30.py — Fig. S11 / Fig. 4f regional MTF/QG small-world metrics,
rebuilt with q=30 (node-count-matched) instead of the original per-channel
q=13 (choose_q_campanharo) construction.

Matches lib.regions._region_metrics_one exactly except:
  - q is fixed at 30 (not choose_q_campanharo(T)=13), matching the main
    q=30 robustness fix applied elsewhere in this package.
  - n_rand is reduced to 10 (from 20) for this supplementary robustness
    check specifically, given the much higher per-trial cost of the
    per-channel construction (~60 individual graph computations per
    trial vs. 2 in the pooled Part 1 analysis). This only affects the
    precision of the randomized-reference null-model average, not the
    correctness of the estimate.
  - Spearman regional rows are UNCHANGED by this fix (Spearman doesn't
    depend on q), so they are copied directly from the existing
    region_trial_metrics.csv cache rather than recomputed, saving
    substantial time.

Resumable: writes in append-mode batches and skips (subject, trial)
pairs already present, so it is safe to invoke repeatedly across
multiple bounded-time calls until complete.

Usage:
    python3 lib/regions_q30.py --limit 400
"""
import sys, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import CACHE, N_RANDOMIZATIONS
from lib.regions import REGION_MAP, _mtf_1d, _choose_k_1d
from lib.metrics import metrics_from_matrix
from lib.representations import load_raw_trial_csv, parse_subject_trial, _list_trials
from lib.config import CH

COLIDX = {c: i for i, c in enumerate(CH)}
Q30 = 30
N_RAND_Q30 = 10
OUT_F = CACHE / "region_trial_metrics_q30.csv"


def _region_metrics_one_q30(path, emo, n_rand=N_RAND_Q30):
    X = load_raw_trial_csv(path); subj, tr = parse_subject_trial(path)
    T = X.shape[0]
    rows = []
    for region, chans in REGION_MAP.items():
        idx = [COLIDX[c] for c in chans if c in COLIDX]
        if len(idx) == 0:
            continue
        base = dict(subject=subj, emotion=emo, trial=Path(path).stem, region=region,
                    n_region_channels=len(idx), q=Q30)
        mt = [metrics_from_matrix(_mtf_1d(X[:, c], Q30, lag=1), n_rand=n_rand) for c in idx]
        rows.append({**base, "method": "MTF", **{k: np.nanmean([d[k] for d in mt]) for k in mt[0]}})
        qg = [metrics_from_matrix(_mtf_1d(X[:, c], Q30, lag=_choose_k_1d(X[:, c])), n_rand=n_rand) for c in idx]
        rows.append({**base, "method": "QG", **{k: np.nanmean([d[k] for d in qg]) for k in qg[0]}})
    return rows


def _done_keys():
    if not OUT_F.exists():
        return set()
    d = pd.read_csv(OUT_F, usecols=["subject", "trial", "method"])
    d = d[d["method"].isin(["MTF", "QG"])]
    # a trial only counts as done if it has entries for ALL regions x both methods
    n_regions = len(REGION_MAP)
    counts = d.groupby(["subject", "trial"]).size()
    complete = counts[counts >= 2 * n_regions].index
    return set(complete)


def seed_spearman_rows():
    """Copy Spearman regional rows from the original cache once, since
    Spearman is unaffected by the MTF/QG q change."""
    orig = CACHE / "region_trial_metrics.csv"
    if not orig.exists():
        print("  [warn] original region_trial_metrics.csv not found; Spearman rows will be missing.")
        return
    d = pd.read_csv(orig)
    sp = d[d.method == "Spearman"].copy()
    sp["q"] = np.nan  # Spearman has no quantile resolution
    if OUT_F.exists():
        existing = pd.read_csv(OUT_F)
        if (existing.method == "Spearman").any():
            print("  [seed] Spearman rows already present, skipping reseed.")
            return
        combined = pd.concat([existing, sp], ignore_index=True)
        combined.to_csv(OUT_F, index=False)
    else:
        sp.to_csv(OUT_F, index=False)
    print(f"  [seed] copied {len(sp)} Spearman regional rows from original cache.")


def run(limit=None, batch=20):
    seed_spearman_rows()
    items = _list_trials()
    done = _done_keys()
    todo = [(p, emo) for p, subj, emo in items
            if (parse_subject_trial(p)[0], Path(p).stem) not in done]
    print(f"[regions_q30] total {len(items)} | done(MTF/QG) {len(done)} | todo {len(todo)}", flush=True)
    if limit is not None:
        todo = todo[:limit]
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        flat = []
        for p, emo in chunk:
            flat.extend(_region_metrics_one_q30(p, emo))
        pd.DataFrame(flat).to_csv(OUT_F, mode="a", header=not OUT_F.exists(), index=False)
        print(f"  +{len(chunk)} trials ({i+len(chunk)}/{len(todo)} this call, {time.time()-t0:.0f}s)", flush=True)
    print(f"[regions_q30] this call processed {len(todo)} trials in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()
    run(limit=a.limit, batch=a.batch)
