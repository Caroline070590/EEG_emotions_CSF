"""Resume fixed, checkpointed null maps after user-interrupted execution.

No additional permutations are selected; completed entries remain unchanged.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from run_session_control import permutation_fit

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
summary = json.loads((OUT / "SESSION_CONTROL_SUMMARY.json").read_text())
assert hashlib.sha256((HERE / "run_session_control.py").read_bytes()).hexdigest() == summary["script_sha256"]
data = np.load(OUT / "primary_feature_snapshot.npz")
frame = pd.read_csv(OUT / "aligned_out_of_fold_predictions.csv")
assert np.array_equal(data["video"], frame.video.values)
groups = frame.source.values
folds = [(str(s), np.flatnonzero(groups != s), np.flatnonzero(groups == s)) for s in np.unique(groups)]
maps = np.load(OUT / "restricted_permutation_clip_labels.npy")
null = np.load(OUT / "restricted_null_balanced_accuracy_progress.npy")
assert len(null) == len(maps) == summary["requested_permutations"] == 1000
completed = np.isfinite(null)
initial_scores = null.copy()
indices = np.flatnonzero(~completed)
provenance = {"checkpoint_entries_retained": int(completed.sum()),
              "remaining_entries": len(indices),
              "fixed_label_maps_sha256": hashlib.sha256((OUT / "restricted_permutation_clip_labels.npy").read_bytes()).hexdigest(),
              "resume_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "initial_checkpoint_sha256": hashlib.sha256((OUT / "restricted_null_balanced_accuracy_progress.npy").read_bytes()).hexdigest()}
(OUT / "RESUME_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(provenance, flush=True)
started = time.monotonic()
jobs = (delayed(permutation_fit)(i, maps[i], data["normalized"], data["video"], folds) for i in indices)
results = Parallel(n_jobs=6, return_as="generator_unordered")(jobs)
for done, (i, score) in enumerate(results, 1):
    null[i] = score
    if done % 20 == 0 or done == len(indices):
        assert np.array_equal(null[completed], initial_scores[completed])
        np.save(OUT / "restricted_null_balanced_accuracy_progress.npy", null)
        print(f"restricted permutations {completed.sum() + done}/1000; resume elapsed {time.monotonic() - started:.1f}s", flush=True)
assert np.isfinite(null).all()
assert np.array_equal(null[completed], initial_scores[completed])
np.save(OUT / "restricted_null_balanced_accuracy.npy", null)
pd.DataFrame({"permutation": np.arange(len(null)), "balanced_accuracy": null}).to_csv(
    OUT / "restricted_null_balanced_accuracy.csv", index=False)
obs = summary["observed_balanced_accuracy"]
summary.update({"analysis_status": "complete", "completed_permutations": len(null),
                "restricted_null_mean": float(null.mean()),
                "restricted_null_95th_percentile": float(np.percentile(null, 95)),
                "restricted_null_exceedances": int((null >= obs).sum()),
                "restricted_null_p_plus_one": float((1 + (null >= obs).sum()) / (1 + len(null))),
                "gain_over_restricted_null_mean": float(obs - null.mean()),
                "resumed_from_checkpoint_entries": int(completed.sum()),
                "resume_elapsed_seconds": time.monotonic() - started,
                "resume_provenance": provenance})
(OUT / "SESSION_CONTROL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ["analysis_status", "observed_balanced_accuracy", "restricted_null_mean", "restricted_null_95th_percentile", "restricted_null_p_plus_one", "resume_elapsed_seconds"]}), flush=True)
