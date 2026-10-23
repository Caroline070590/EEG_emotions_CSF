"""Build a float32 memmap of SEED-VII trials in the FACED format (1600 x 30 x 7500).

Official preprocessed EEG (EEG_preprocessed/<subject>.mat; keys '1'..'80', 62 x n at 200 Hz).
For each clip: the 30 FACED channels, the last 30 s (6000 samples), resampled 200 -> 250 Hz
(polyphase, up 5 / down 4) -> 30 x 7500, identical to the FACED trials.
Stimulus metadata: clip index (1-80), condition, session, and source program (clips that are
excerpts of the same film or programme share a source; used for leave-one-source-out decoding).

usage: 00_load_clips.py SEED7_DIR      (default ~/Data/SEED-VII)
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parents[1]; OUT = HERE / "cache"
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "Data" / "SEED-VII")
CH = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'CZ', 'C3', 'C4',
      'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'PZ', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'OZ', 'O1', 'O2']
FS_IN, SEC, T = 200, 30, 7500
LABEL = {0: "disgust", 1: "fear", 2: "sadness", 3: "neutral", 4: "happiness", 5: "anger", 6: "surprise"}


def clip_table():
    lab = pd.read_excel(SRC / "emotion_label_and_stimuli_order.xlsx", header=None)
    rows = []
    for s in range(4):
        idx, emo = lab.iloc[2 * s, 1:21].astype(int).values, lab.iloc[2 * s + 1, 1:21].values
        st = pd.read_excel(SRC / "SEED-VII_stimulation.xlsx", sheet_name=f"experiment {s + 1}", header=0).iloc[:20]
        for k in range(20):
            name = str(st.iloc[k, 0]); code = int(st.iloc[k, 1])
            assert LABEL[code] == {"Happy": "happiness", "Sad": "sadness"}.get(emo[k], emo[k].lower()), (idx[k], name, emo[k])
            # source programme: English title before '/', lower-case, episode / season tags removed
            src = re.sub(r"\b(ep|episode|season|s)\s*\d+\b.*", "", name.split("/")[0].lower())
            src = re.sub(r"[^a-z0-9 ]", " ", src).split()
            rows.append({"clip": idx[k], "session": s + 1, "emotion": LABEL[code], "title": name,
                         "source": " ".join(src[:4]) if src else name})
    c = pd.DataFrame(rows).sort_values("clip").reset_index(drop=True)
    assert list(c["clip"]) == list(range(1, 81))
    return c


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    clips = clip_table()
    clips.to_csv(OUT / "clips.csv", index=False)
    order = [str(x).strip().upper() for x in pd.read_excel(SRC / "Channel Order.xlsx", header=None)[0]]
    assert len(order) == 62 and all(c in order for c in CH), set(CH) - set(order)
    sel = [order.index(c) for c in CH]
    subs = list(range(1, 21))
    meta = pd.DataFrame([{"subject": s, "video": v - 1, "clip": v} for s in subs for v in range(1, 81)])
    meta = meta.merge(clips, on="clip", how="left")
    meta.to_csv(OUT / "meta.csv", index_label="idx")
    mm = np.lib.format.open_memmap(OUT / "eeg_30s.npy", mode="w+", dtype=np.float32, shape=(len(meta), 30, T))
    lens = []
    for i, s in enumerate(subs):
        m = sio.loadmat(SRC / "EEG_preprocessed" / f"{s}.mat")
        for v in range(1, 81):
            x = m[str(v)]
            assert x.shape[0] == 62 and x.shape[1] >= SEC * FS_IN, (s, v, x.shape)
            lens.append(x.shape[1] / FS_IN)
            y = resample_poly(x[sel, -SEC * FS_IN:].astype(np.float64), 5, 4, axis=1)
            assert y.shape == (30, T)
            mm[i * 80 + v - 1] = y
        del m
        print(f"subject {s} done", flush=True)
    mm.flush()
    print("done", mm.shape, "clip length (s): min %.0f median %.0f max %.0f" % (min(lens), np.median(lens), max(lens)))
    print(clips.groupby("emotion").size().to_dict(), "| sources:", clips.source.nunique())
