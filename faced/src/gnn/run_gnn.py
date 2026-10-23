"""Leave-one-video-out evaluation of SD-GNN and baselines on simulated or real data.

Examples
  python run_gnn.py --data sim --delta 0.2 --sigma-video 1.0 --sim-seed 1 --model sdgnn --lam 1.0
  python run_gnn.py --data real --model sdgnn --lam 1.0 --perm 17      # one permutation-null run
Writes one JSON line per run to --out.

Simulation (same layout as FACED: 123 participants x 28 videos, same video->condition map):
  inputs = participant offset + video fingerprint (sd sigma_video, on every feature)
           + condition effect (each non-neutral condition: its own 20-edge subnetwork in the
             planted view, effect size delta in noise-SD units) + trial noise (sd 1).
  The condition effect is shared by all videos of a condition; video fingerprints are not.
  delta = 0 gives a null simulation with no condition information at all.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from sdgnn import (load_real, lovo, balanced_acc, zscore_within, edges_to_adj, VIEWS, EMOS, IU)

CACHE = Path(__file__).resolve().parents[2] / "cache"
VIDEO_TO_CLS = np.array([0]*3 + [1]*3 + [2]*3 + [3]*3 + [4]*4 + [5]*3 + [6]*3 + [7]*3 + [8]*3)
PLANT_VIEW = VIEWS.index("oaec_alpha")


def simulate(delta, sigma_video, seed, n_subj=123, n_edges=20):
    rng = np.random.default_rng(seed)
    V = len(VIEWS); S, NV = n_subj, 28
    subj = np.repeat(np.arange(S), NV); video = np.tile(np.arange(NV), S); y = VIDEO_TO_CLS[video]
    n = len(subj)
    # node features (power): participant offsets + video fingerprints + noise
    X = rng.normal(0, 1, (S, 30, 5))[subj] \
        + rng.normal(0, sigma_video, (NV, 30, 5))[video] + rng.normal(0, 1, (n, 30, 5))
    # edges
    E = rng.normal(0, 1, (S, V, 435))[subj] + rng.normal(0, sigma_video, (NV, V, 435))[video] \
        + rng.normal(0, 1, (n, V, 435))
    # condition-specific subnetworks in the planted view (neutral = no effect)
    for c in range(9):
        if EMOS[c] == "neutral": continue
        idx = rng.choice(435, n_edges, replace=False); sign = rng.choice([-1.0, 1.0], n_edges)
        m = y == c
        E[np.ix_(m, [PLANT_VIEW], idx)] += delta * sign
    X = zscore_within(X.astype(np.float32), subj)
    A = np.stack([edges_to_adj(zscore_within(E[:, v].astype(np.float32), subj)) for v in range(V)], 1)
    return dict(X=X, A=A, y=y, video=video, subject=subj, views=VIEWS)


CALIB = CACHE / "gnn_sim_calibration.npz"

def make_calibration():
    """Per-feature condition-mean patterns (relative to neutral) and between-video-within-
    condition SD, estimated from the real within-participant z-scored features."""
    d = load_real(CACHE); N = len(d["y"]); video = d["video"]; y = d["y"]
    F_ = np.concatenate([d["X"].reshape(N, -1), d["A"][:, :, IU[0], IU[1]].reshape(N, -1)], 1)
    vm = np.stack([F_[video == v].mean(0) for v in range(28)]); v2c = VIDEO_TO_CLS
    cm = np.stack([vm[v2c == c].mean(0) for c in range(9)])
    within = np.concatenate([vm[v2c == c] - cm[c] for c in range(9)])
    sig_v = np.sqrt(np.clip(within.var(0) * 28 / 19 - 1 / 123, 0, None))
    np.savez(CALIB, effect=(cm - cm[EMOS.index("neutral")]).astype(np.float32), sigma_video=sig_v.astype(np.float32))

def simulate_calibrated(scale, kappa, seed):
    """Semi-synthetic data calibrated to FACED: trial = scale * real condition pattern
    + video fingerprint (kappa * real between-video SD per feature) + noise (SD 1)."""
    if not CALIB.exists(): make_calibration()
    c = np.load(CALIB); eff, sv = c["effect"], c["sigma_video"]; D = eff.shape[1]
    rng = np.random.default_rng(seed); S, NV = 123, 28
    subj = np.repeat(np.arange(S), NV); video = np.tile(np.arange(NV), S); y = VIDEO_TO_CLS[video]
    F_ = scale * eff[y] + (kappa * sv * rng.normal(0, 1, (NV, D)).astype(np.float32))[video] \
         + rng.normal(0, 1, (len(y), D)).astype(np.float32)
    V = len(VIEWS)
    X = zscore_within(F_[:, :150].reshape(-1, 30, 5), subj)
    E = F_[:, 150:].reshape(-1, V, 435)
    A = np.stack([edges_to_adj(zscore_within(E[:, v], subj)) for v in range(V)], 1)
    return dict(X=X, A=A, y=y, video=video, subject=subj, views=VIEWS)


REAL_STRUCT = CACHE / "gnn_sim_realstruct.npz"

def make_realstruct():
    """Amendment 4: between-video-within-condition covariance (low rank, noise-corrected) and
    real trial residuals (trial minus its video mean), from the real z-scored features."""
    d = load_real(CACHE); N = len(d["y"]); video = d["video"]
    F_ = np.concatenate([d["X"].reshape(N, -1), d["A"][:, :, IU[0], IU[1]].reshape(N, -1)], 1)
    vm = np.stack([F_[video == v].mean(0) for v in range(28)]); v2c = VIDEO_TO_CLS
    within = np.concatenate([vm[v2c == c] - vm[v2c == c].mean(0) for c in range(9)])
    _, S, Vt = np.linalg.svd(within, full_matrices=False)
    lam = np.clip(S ** 2 / 19 - 1 / 123, 0, None)        # noise-corrected eigenvalues (df = 28 - 9)
    keep = lam > 0
    np.savez(REAL_STRUCT, fp_basis=(np.sqrt(lam[keep])[:, None] * Vt[keep]).astype(np.float32),
             residual=(F_ - vm[video]).astype(np.float32))

def simulate_realstruct(scale, kappa, seed):
    """Like simulate_calibrated, but video fingerprints have the real cross-feature covariance and
    trial noise is resampled from real trial residuals (both correlated across features)."""
    if not CALIB.exists(): make_calibration()
    if not REAL_STRUCT.exists(): make_realstruct()
    eff = np.load(CALIB)["effect"]; r = np.load(REAL_STRUCT); B, res = r["fp_basis"], r["residual"]
    rng = np.random.default_rng(seed); S, NV = 123, 28
    subj = np.repeat(np.arange(S), NV); video = np.tile(np.arange(NV), S); y = VIDEO_TO_CLS[video]
    fp = kappa * rng.normal(0, 1, (NV, len(B))).astype(np.float32) @ B
    F_ = scale * eff[y] + fp[video] + res[rng.integers(0, len(res), len(y))]
    V = len(VIEWS)
    X = zscore_within(F_[:, :150].reshape(-1, 30, 5), subj)
    E = F_[:, 150:].reshape(-1, V, 435)
    A = np.stack([edges_to_adj(zscore_within(E[:, v], subj)) for v in range(V)], 1)
    return dict(X=X, A=A, y=y, video=video, subject=subj, views=VIEWS)


REGION_MAP = {   # the paper's regional grouping
    "frontal": ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"],
    "central": ["FC1", "FC2", "FC5", "FC6", "CZ", "C3", "C4"],
    "temporal": ["T7", "T8"],
    "parietal": ["CP1", "CP2", "CP5", "CP6", "P3", "P4", "P7", "P8", "PZ"],
    "occipital": ["PO3", "PO4", "O1", "O2", "OZ"]}
HOMOLOGUES = [("FP1", "FP2"), ("F3", "F4"), ("F7", "F8"), ("FC1", "FC2"), ("FC5", "FC6"), ("C3", "C4"),
              ("T7", "T8"), ("CP1", "CP2"), ("CP5", "CP6"), ("P3", "P4"), ("P7", "P8"), ("PO3", "PO4"), ("O1", "O2")]
CHANNELS = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ', 'C3', 'C4', 'T7', 'T8',
            'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2']

def prior_adjacency(kind):
    """Amendment 5: fixed graph priors. Symmetrically degree-normalized and scaled by sqrt(30) to
    offset the 1/sqrt(30) of the message-passing layer."""
    ix = {c: i for i, c in enumerate(CHANNELS)}; A = np.zeros((30, 30))
    if kind == "biology":
        for chans in REGION_MAP.values():
            for a in chans:
                for b in chans:
                    if a != b: A[ix[a], ix[b]] = 1.0
        for a, b in HOMOLOGUES: A[ix[a], ix[b]] = A[ix[b], ix[a]] = 1.0
    elif kind == "physics":
        import mne
        pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        name = {k.upper(): k for k in pos}
        P = np.array([pos[name[c]] for c in CHANNELS])
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        A = np.exp(-(D / np.median(D[np.triu_indices(30, 1)])) ** 2); np.fill_diagonal(A, 0)
    elif kind != "none":
        raise ValueError(kind)
    if A.any():
        d = A.sum(1); A = A / np.sqrt(np.outer(d, d)) * np.sqrt(30)
    return A.astype(np.float32)


def sync_mask(kind):
    """Amendment 6: binary mask on the measured synchronization edges."""
    if kind == "full":
        return np.ones((30, 30), np.float32) - np.eye(30, dtype=np.float32)
    reg = {c: r for r, cs in REGION_MAP.items() for c in cs}
    M = np.array([[float(reg[a] == reg[b] and a != b) for b in CHANNELS] for a in CHANNELS], np.float32)
    if kind == "random":   # same number of edges, uniformly at random, fixed seed
        n = int(M[np.triu_indices(30, 1)].sum()); iu = np.triu_indices(30, 1)
        pick = np.random.default_rng(2026).choice(len(iu[0]), n, replace=False)
        M = np.zeros((30, 30), np.float32); M[iu[0][pick], iu[1][pick]] = 1; M = M + M.T
    elif kind != "biology":
        raise ValueError(kind)
    return M


def lda_lovo(data, video_to_cls):
    F_ = np.concatenate([data["X"].reshape(len(data["y"]), -1),
                         data["A"][:, :, IU[0], IU[1]].reshape(len(data["y"]), -1)], 1)
    y = video_to_cls[data["video"]]; pred = np.empty_like(y)
    for v in range(28):
        te = data["video"] == v; tr = ~te
        sc = StandardScaler().fit(F_[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(F_[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(F_[te]))
    return pred


def lda_summary_lovo(data, video_to_cls):
    """Shrinkage LDA on the global summaries only (mean edge weight per view + channel-mean node features)."""
    S = np.concatenate([data["A"][:, :, IU[0], IU[1]].mean(-1), data["X"].mean(1)], 1)
    y = video_to_cls[data["video"]]; pred = np.empty_like(y)
    for v in range(28):
        te = data["video"] == v; tr = ~te
        sc = StandardScaler().fit(S[tr])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(S[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(S[te]))
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["sim", "simcal", "simreal", "real"], required=True)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--sigma-video", type=float, default=1.0)
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="simcal: multiple of the real condition pattern")
    ap.add_argument("--kappa", type=float, default=1.0, help="simcal: multiple of the real between-video SD")
    ap.add_argument("--model", choices=["sdgnn", "cgnn", "hybrid", "sfgnn", "mlp", "lda", "lda_summary"], default="sdgnn")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--prior", choices=["none", "physics", "biology"], default=None,
                    help="Amendment 5: replace the data-derived edge views by one fixed graph prior")
    ap.add_argument("--sync-mask", choices=["full", "biology", "random"], default=None,
                    help="Amendment 6: keep only the wPLI/oAEC views and mask their edges")
    ap.add_argument("--rex", type=float, default=0.0, help="V-REx penalty weight (0 = off)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--perm", type=int, default=0, help="0 = observed labels; k>0 = permutation k of the video->condition map")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--recurrence", action="store_true", help="real data: add RQA node features and the joint-recurrence view")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    t0 = time.time()
    data = (load_real(CACHE, recurrence=a.recurrence) if a.data == "real" else
            simulate_calibrated(a.scale, a.kappa, a.sim_seed) if a.data == "simcal" else
            simulate_realstruct(a.scale, a.kappa, a.sim_seed) if a.data == "simreal" else
            simulate(a.delta, a.sigma_video, a.sim_seed))
    if a.prior:
        data["A"] = np.repeat(prior_adjacency(a.prior)[None, None], len(data["y"]), 0)
        data["views"] = [f"prior_{a.prior}"]
    if a.sync_mask:
        keep = [i for i, v in enumerate(data["views"]) if v.startswith(("wpli_", "oaec_"))]
        assert len(keep) == 10, data["views"]
        data["A"] = np.ascontiguousarray(data["A"][:, keep] * sync_mask(a.sync_mask)[None, None])
        data["views"] = [data["views"][i] for i in keep]
    v2c = VIDEO_TO_CLS.copy()
    if a.perm > 0:
        v2c = np.random.default_rng(10_000 + a.perm).permutation(VIDEO_TO_CLS)
    y = v2c[data["video"]]
    if a.model == "lda":
        pred = lda_lovo(data, v2c)
    elif a.model == "lda_summary":
        pred = lda_summary_lovo(data, v2c)
    else:
        pred = lovo(data, v2c, lam=(a.lam if a.model in ("sdgnn", "cgnn", "hybrid", "sfgnn") else 0.0), model=a.model,
                    epochs=a.epochs, rex=a.rex)
    res = dict(vars(a), balanced_accuracy=balanced_acc(y, pred), seconds=round(time.time() - t0, 1))
    if a.data == "real" and a.perm == 0:
        res["per_video_accuracy"] = [float(np.mean(pred[data["video"] == v] == y[data["video"] == v])) for v in range(28)]
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as fh: fh.write(json.dumps(res) + "\n")
