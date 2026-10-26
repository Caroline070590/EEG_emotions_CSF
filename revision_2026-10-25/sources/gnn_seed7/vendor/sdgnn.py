"""Dataset dimension adaptation for the SEED-VII participant-held-out extension.

Copied from participant_prediction_2026-10-24/vendor/sdgnn.py, SHA256
056cad7777309868000da8ec00da6288e196b1164deac06e5ff317f4d7ce0bde.
Only train_predict class/video dimensions, their input validation, and LOVO iteration
are generalized. Architecture definitions, default dimensions, loss terms, optimizer,
and training hyperparameters are retained. Dimensions are explicitly supplied by
train_predict, so the original nine-class/28-video behavior is preserved.

The legacy FACED load_real/zscore_within helpers remain for compatibility; the
SEED-VII runner must instead supply its independently loaded inputs with scaling
fitted on training participants only. The following description is the inherited
architecture documentation.

Stimulus-disentangled multi-view graph network (SD-GNN) for EEG condition decoding.

Each trial is a set of graphs on the same 30 EEG channels:
  node features  X  (30 x 5)   log band power (delta..gamma)
  edge views     A_v (30 x 30) band-specific wPLI and oAEC (5 bands each) and signed Spearman
All inputs are z-scored within participant (label-free), so the model sees deviations from
each participant's own typical power and connectivity.

Encoder: one message-passing layer per view, H_v = GELU(W_v [X, A_v X / sqrt(30)]), mixed by a
learned softmax gate over views; the gated node embeddings are flattened (spatial identity of
channels is meaningful in sensor space) and projected to a low-dimensional code z.

Heads:
  condition head      p(condition | z)
  video adversary     p(video | z, condition), restricted to the videos of that condition and
                      trained through a gradient-reversal layer. The encoder is therefore pushed
                      to remove information that distinguishes videos *within* a condition,
                      while information shared by all videos of a condition is untouched.
                      (An unconditional video adversary would also remove the condition signal,
                      because each video belongs to exactly one condition.)
lambda_adv = 0 gives the same network without disentanglement (ablation).
"""
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
VIEWS = [f"wpli_{b}" for b in BANDS] + [f"oaec_{b}" for b in BANDS] + ["rho_signed"]
EMOS = ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'amusement', 'inspiration', 'joy', 'tenderness']
IU = np.triu_indices(30, 1)


# ------------------------------------------------------------------ data
def zscore_within(x, groups):
    """z-score each feature within group (participant); x: (N, ...)."""
    out = np.empty_like(x, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        mu, sd = x[m].mean(0, keepdims=True), x[m].std(0, keepdims=True) + 1e-6
        out[m] = (x[m] - mu) / sd
    return out

def edges_to_adj(E):
    """(N, 435) upper-triangle vectors -> (N, 30, 30) symmetric, zero diagonal."""
    A = np.zeros((E.shape[0], 30, 30), np.float32)
    A[:, IU[0], IU[1]] = E; A[:, IU[1], IU[0]] = E
    return A

def load_real(cache, recurrence=False):
    meta = pd.read_csv(Path(cache) / "features_scalar.csv", usecols=["idx", "subject", "video", "emotion"])
    arr = np.load(Path(cache) / "features_arrays.npz")
    subj = meta.subject.values
    X = np.stack([arr[f"logpow_{b}"] for b in BANDS], -1)                       # N,30,5
    keys = [f"pair_{v}" if v != "rho_signed" else v for v in VIEWS]
    E = [arr[k] for k in keys]; views = list(VIEWS)
    if recurrence:   # RQA node features (30 x 5) and joint-recurrence synchronization view
        rec = np.load(Path(cache) / "recurrence_features.npz")
        X = np.concatenate([X, rec["rqa"]], -1)                                 # N,30,10
        E.append(rec["jrs"]); views.append("joint_recurrence")
    X = zscore_within(X, subj)
    A = np.stack([edges_to_adj(zscore_within(e, subj)) for e in E], 1)          # N,V,30,30
    y = meta.emotion.map({e: i for i, e in enumerate(EMOS)}).values
    return dict(X=X, A=A, y=y, video=meta.video.values, subject=subj, views=views)


# ------------------------------------------------------------------ model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None

class SDGNN(nn.Module):
    def __init__(self, n_views, n_feat=5, hidden=8, code=16, n_cls=9, n_videos=28, dropout=0.3,
                 gate="softmax"):
        super().__init__()
        self.gate_type = gate   # "softmax" (views compete) or "sigmoid" (constrained: independent [0,1] gates)
        self.msg = nn.ModuleList([nn.Linear(2 * n_feat, hidden) for _ in range(n_views)])
        self.view_logit = nn.Parameter(torch.zeros(n_views))
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(30 * hidden, code), nn.GELU())
        self.cls = nn.Linear(code, n_cls)
        self.adv = nn.Sequential(nn.Linear(code + n_cls, 32), nn.GELU(), nn.Linear(32, n_videos))

    def gates(self):
        if self.gate_type == "sigmoid":
            return torch.sigmoid(self.view_logit)
        return torch.softmax(self.view_logit, 0)

    def encode(self, X, A, view_mask=None):
        g = self.gates()
        if view_mask is not None:                       # used for view attribution
            g = g * view_mask
            if self.gate_type == "softmax": g = g / g.sum().clamp_min(1e-8)
        H = 0
        for v, lin in enumerate(self.msg):
            M = torch.bmm(A[:, v], X) / np.sqrt(30.0)
            H = H + g[v] * F.gelu(lin(torch.cat([X, M], -1)))
        return self.proj(H.flatten(1))

    def forward(self, X, A, view_mask=None):
        z = self.encode(X, A, view_mask)
        return z, self.cls(z)


class HybridGNN(SDGNN):
    """Constrained GNN (sigmoid view gates) + a direct path for global summaries: the mean edge
    weight of every view and the channel-mean of every node feature. The simulations showed that
    message passing alone compresses away dense, global shifts in coupling; this path keeps them.
    The adversary sees the concatenated code."""
    def __init__(self, n_views, n_feat, hidden=8, code=16, summ=8, n_cls=9, n_videos=28, dropout=0.3):
        super().__init__(n_views, n_feat=n_feat, hidden=hidden, code=code, n_cls=n_cls,
                         n_videos=n_videos, dropout=dropout, gate="sigmoid")
        self.summ = nn.Sequential(nn.Linear(n_views + n_feat, summ), nn.GELU())
        self.cls = nn.Linear(code + summ, n_cls)
        self.adv = nn.Sequential(nn.Linear(code + summ + n_cls, 32), nn.GELU(), nn.Linear(32, n_videos))

    def forward(self, X, A, view_mask=None):
        g = A[:, :, IU[0], IU[1]].mean(-1)                      # N x V  mean edge weight per view
        z = torch.cat([self.encode(X, A, view_mask), self.summ(torch.cat([g, X.mean(1)], 1))], 1)
        return z, self.cls(z)


class SummaryFirstGNN(SDGNN):
    """Summary-first model (Amendment 4): a linear head on the batch-normalised global summaries
    carries the prediction; the constrained GNN can only add a residual, scaled by a scalar alpha
    that starts at 0 and is shrunk by an L1 penalty."""
    def __init__(self, n_views, n_feat, hidden=8, code=16, n_cls=9, n_videos=28, dropout=0.3):
        super().__init__(n_views, n_feat=n_feat, hidden=hidden, code=code, n_cls=n_cls,
                         n_videos=n_videos, dropout=dropout, gate="sigmoid")
        self.norm = nn.BatchNorm1d(n_views + n_feat, affine=False)
        self.lin = nn.Linear(n_views + n_feat, n_cls)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.adv = nn.Sequential(nn.Linear(n_views + n_feat + code + n_cls, 32), nn.GELU(), nn.Linear(32, n_videos))

    def forward(self, X, A, view_mask=None):
        s = self.norm(torch.cat([A[:, :, IU[0], IU[1]].mean(-1), X.mean(1)], 1))
        zg = self.encode(X, A, view_mask)
        return torch.cat([s, zg], 1), self.lin(s) + self.alpha * self.cls(zg)


class MLP(nn.Module):
    """No-graph baseline on the same inputs (flattened power + upper-triangle edges)."""
    def __init__(self, d_in, code=16, n_cls=9, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, code), nn.GELU(), nn.Linear(code, n_cls))
    def forward(self, X, A, view_mask=None):
        f = torch.cat([X.flatten(1), A[:, :, IU[0], IU[1]].flatten(1)], 1)
        return None, self.net(f)


# ------------------------------------------------------------------ training
def train_predict(data, tr, te, video_to_cls, lam=1.0, model="sdgnn", epochs=40, lr=3e-3, wd=1e-3,
                  batch=128, seed=0, return_model=False, rex=0.0):
    """Fit on `tr`, predict `te`; video_to_cls contains one label for each video ID.
    Labels must be contiguous integers 0..n_cls-1; video IDs index this mapping.
    Class and adversary output dimensions follow this map (including under label
    permutation). rex > 0 adds the inherited V-REx penalty. Returns predicted labels
    and, when requested, the fitted model. Caller supplies already scaled inputs.
    """
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X = torch.from_numpy(data["X"]); A = torch.from_numpy(data["A"])
    video_to_cls = np.asarray(video_to_cls)
    if (video_to_cls.ndim != 1 or not len(video_to_cls)
            or not np.issubdtype(video_to_cls.dtype, np.integer)):
        raise ValueError("video_to_cls must be a nonempty integer vector")
    classes = np.unique(video_to_cls)
    n_cls, n_videos = len(classes), len(video_to_cls)
    if not np.array_equal(classes, np.arange(n_cls)):
        raise ValueError("video_to_cls labels must be contiguous integers starting at zero")
    video_to_cls = video_to_cls.astype(np.int64, copy=False)
    vid = np.asarray(data["video"])
    if (vid.ndim != 1 or len(vid) != len(X) or not len(vid)
            or not np.issubdtype(vid.dtype, np.integer)
            or vid.min() < 0 or vid.max() >= n_videos):
        raise ValueError("Every trial video ID must index video_to_cls")
    vid = vid.astype(np.int64, copy=False)
    y = video_to_cls[vid]
    if model == "sdgnn":
        net = SDGNN(A.shape[1], n_feat=X.shape[-1], n_cls=n_cls, n_videos=n_videos)
    elif model == "cgnn":   # constrained GNN (design of the sleep-condition project): sigmoid gates + fixed L1 shrinkage
        net = SDGNN(A.shape[1], n_feat=X.shape[-1], n_cls=n_cls, n_videos=n_videos, gate="sigmoid")
    elif model == "hybrid":
        net = HybridGNN(A.shape[1], n_feat=X.shape[-1], n_cls=n_cls, n_videos=n_videos)
    elif model == "sfgnn":
        net = SummaryFirstGNN(A.shape[1], n_feat=X.shape[-1], n_cls=n_cls, n_videos=n_videos)
    else:
        net = MLP(30 * X.shape[-1] + A.shape[1] * 435, n_cls=n_cls)
    SHRINK = 1e-2           # fixed shrinkage on view gates (not tuned)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    # Inherited class-balanced loss, sized to the supplied condition map.
    cnt = np.bincount(y[tr], minlength=n_cls).astype(np.float32)
    w = torch.from_numpy(np.where(cnt > 0, cnt.sum() / (n_cls * np.maximum(cnt, 1)), 0).astype(np.float32))
    # conditional adversary mask: for each class, which training videos belong to it
    train_videos = np.unique(vid[tr])
    vmask = torch.full((n_cls, n_videos), -1e9)
    for v in train_videos: vmask[video_to_cls[v], v] = 0.0
    v2c_t = torch.from_numpy(np.asarray(video_to_cls))
    nvid = torch.bincount(v2c_t[torch.from_numpy(train_videos)], minlength=n_cls).float().clamp_min(1)
    yt, vt = torch.from_numpy(y), torch.from_numpy(vid)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), batch):
            b = torch.from_numpy(perm[i:i + batch])
            z, logit = net(X[b], A[b])
            if rex > 0:      # per-video mean loss, then class-balanced mean + rex * variance across videos
                # No per-trial class weights here: per-video averaging plus class weights would
                # compensate twice and bias the model towards the class with fewest training
                # videos, which under leave-one-video-out is always the held-out class.
                l = F.cross_entropy(logit, yt[b], reduction="none")
                uv, inv = torch.unique(vt[b], return_inverse=True)
                per = torch.zeros(len(uv)).index_add_(0, inv, l) / torch.bincount(inv).float()
                vw = 1.0 / nvid[v2c_t[uv]]; vw = vw / vw.sum()
                loss = (vw * per).sum() + rex * per.var()
            else:
                loss = F.cross_entropy(logit, yt[b], weight=w)
            if model in ("cgnn", "hybrid", "sfgnn"):
                loss = loss + SHRINK * net.gates().sum()
            if model == "sfgnn":
                loss = loss + SHRINK * net.alpha.abs().sum()
            if model in ("sdgnn", "cgnn", "hybrid", "sfgnn") and lam > 0:
                zr = GradReverse.apply(z, lam)
                onehot = F.one_hot(yt[b], n_cls).float()
                adv = net.adv(torch.cat([zr, onehot], 1)) + vmask[yt[b]]
                loss = loss + F.cross_entropy(adv, vt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        _, logit = net(X[torch.from_numpy(te)], A[torch.from_numpy(te)])
    pred = logit.argmax(1).numpy()
    return (pred, net) if return_model else pred


def lovo(data, video_to_cls, **kw):
    """Leave-one-video-out predictions for every trial."""
    pred = np.empty(len(data["y"]), int)
    for v in np.unique(data["video"]):
        te = np.where(data["video"] == v)[0]; tr = np.where(data["video"] != v)[0]
        pred[te] = train_predict(data, tr, te, video_to_cls, seed=v, **kw)
    return pred


def balanced_acc(y, p, n_cls=9):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(n_cls) if (y == c).any()]))
