"""Demo: stimulus-aware inference and decoding of unseen stimuli on simulated data (runs in under a minute).

Simulates a FACED-like design (40 participants x 28 videos; 8 affective conditions with 3 videos each and a
neutral condition with 4 videos) and two features:

* feature A, a condition effect shared by all videos of a condition (generalizes to new stimuli);
* feature B, video-specific fingerprints only (no condition effect beyond the particular clips).

It then applies the two core procedures of the paper:

1. emotion-versus-neutral tests with participants random only, and with participants AND videos as crossed
   random factors (min F'; faced/src/lib_rev.py);
2. leave-one-video-out decoding with a stimulus-structured permutation null (the video-to-condition map is
   permuted, so the null keeps the stimulus structure).

Expected behaviour: feature B looks significant when videos are fixed but not when videos are random, and
does not decode unseen videos; feature A survives both. See expected_output.txt.

usage: python demo/run_demo.py
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "faced" / "src"))
from lib_rev import contrast_vs_neutral, EMOS  # noqa: E402

t0 = time.time(); rng = np.random.default_rng(7)
N_SUBJ = 40
videos = [(v, e) for v, e in enumerate([e for e in EMOS if e != "neutral" for _ in range(3)] + ["neutral"] * 4)]
cond_effect = {e: (0 if e == "neutral" else rng.normal(-0.6, 0.25)) for e in EMOS}
video_fp = rng.normal(0, 0.9, len(videos))
rows = []
for s in range(N_SUBJ):
    subj_offset = rng.normal(0, 1, 2)
    for v, e in videos:
        rows.append({"subject": s, "video": v, "emotion": e,
                     "feature_A_condition": subj_offset[0] + cond_effect[e] + 0.25 * video_fp[v] + rng.normal(0, 1),
                     "feature_B_video_only": subj_offset[1] + video_fp[v] + rng.normal(0, 1)})
df = pd.DataFrame(rows)
print(f"Simulated {N_SUBJ} participants x {len(videos)} videos = {len(df)} trials\n")

print("1. Emotion-versus-neutral tests (p < 0.05 out of 8 contrasts)")
for f in ("feature_A_condition", "feature_B_video_only"):
    res = pd.DataFrame([contrast_vs_neutral(df[f].to_numpy(), df, e) for e in EMOS if e != "neutral"])
    print(f"   {f:22s} participants random only: {(res.p1 < .05).sum()}/8   participants + videos random: {(res.p_minF < .05).sum()}/8")

print("\n2. Decoding the condition of unseen videos (leave-one-video-out, 200 permutations)")
emo_idx = {e: i for i, e in enumerate(EMOS)}
v2e = np.array([e for _, e in videos]); video = df.video.to_numpy()
def lovo(X, mapping):
    y = np.array([emo_idx[mapping[v]] for v in video]); pred = np.empty_like(y)
    for v in range(len(videos)):
        te = video == v
        pred[te] = LinearDiscriminantAnalysis().fit(X[~te], y[~te]).predict(X[te])
    return balanced_accuracy_score(y, pred)
for f in ("feature_A_condition", "feature_B_video_only"):
    X = df.groupby("subject")[f].transform(lambda x: (x - x.mean()) / x.std()).to_numpy()[:, None]
    obs = lovo(X, v2e); null = np.array([lovo(X, rng.permutation(v2e)) for _ in range(200)])
    p = (1 + (null >= obs).sum()) / 201
    print(f"   {f:22s} accuracy {100*obs:4.1f}%   null mean {100*null.mean():4.1f}%   p = {p:.3f}")
print(f"\nCompleted in {time.time() - t0:.0f} s")
