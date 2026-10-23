"""Summarize a paired-graph experiment (Amendments 5 and 6): each condition's gain over its own
stimulus-structured null and paired one-sided contrasts against the paired permutation differences.
usage: summarize_paired.py RESULTS.jsonl KEY OUT.csv CONTRAST_A-CONTRAST_B ..."""
import json, sys
import numpy as np, pandas as pd
f, key, out, contrasts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
r = pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates([key, "perm"], keep="last")
r["acc"] = r.balanced_accuracy * 100
w = r.pivot(index="perm", columns=key, values="acc")
obs, null = w.loc[0], w.drop(index=0)
rows = []
for k in w.columns:
    n = null[k].dropna()
    rows.append(dict(test=f"{k} vs own null", observed=obs[k], null_mean=n.mean(), gain=obs[k] - n.mean(),
                     p=(1 + (n >= obs[k]).sum()) / (1 + len(n)), n_perm=len(n)))
for c in contrasts:
    a, b = c.split("-")
    d = (null[a] - null[b]).dropna(); o = obs[a] - obs[b]
    rows.append(dict(test=f"{a} minus {b} (paired)", observed=o, null_mean=d.mean(), gain=o - d.mean(),
                     p=(1 + (d >= o).sum()) / (1 + len(d)), n_perm=len(d)))
T = pd.DataFrame(rows); T.to_csv(out, index=False)
print(T.round(3).to_string(index=False))
