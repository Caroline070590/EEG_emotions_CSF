"""Build a float32 memmap of the complete 30-s FACED trials (3444 x 30 x 7500)
from EEG-data-raw-multiscale.zip (full_trial export). trial_id == FACED video id."""
import zipfile, io, gzip, sys, os
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed

OUT = Path(__file__).resolve().parents[1] / "cache"
OUT.mkdir(parents=True, exist_ok=True)
# path to the FACED full-trial export; override with the FACED_ZIP environment variable
ZIP = Path(os.environ.get("FACED_ZIP", Path(__file__).resolve().parents[1] / "data" / "EEG-data-raw-multiscale.zip"))
CH = ['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4',
      'T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']
T = 7500

def load(rows):
    z = zipfile.ZipFile(ZIP); out = []
    for i, f in rows:
        df = pd.read_csv(io.BytesIO(gzip.decompress(z.read(f))), usecols=CH)
        X = df[CH].to_numpy(np.float32).T
        assert X.shape == (30, T), (f, X.shape)
        out.append((i, X))
    return out

if __name__ == "__main__":
    z = zipfile.ZipFile(ZIP)
    m = pd.read_csv(z.open("EEG-data-raw-multiscale/metadata/manifest_full_trial.csv"))
    m = m.sort_values(["subject_id", "trial_id"]).reset_index(drop=True)
    assert len(m) == 3444 and (m.n_timepoints == T).all()
    meta = pd.DataFrame({"subject": m.subject_id, "video": m.trial_id,
                         "emotion": m.emotion_name, "valence": m.valence, "file": m.csv_file})
    meta.to_csv(OUT / "meta.csv", index_label="idx")
    mm = np.lib.format.open_memmap(OUT / "eeg_30s.npy", mode="w+", dtype=np.float32, shape=(len(m), 30, T))
    chunks = np.array_split(np.arange(len(m)), 96)
    for res in Parallel(n_jobs=12, verbose=2)(delayed(load)([(i, m.csv_file[i]) for i in c]) for c in chunks):
        for i, X in res: mm[i] = X
    mm.flush()
    print("done", mm.shape, float(np.abs(mm[::200]).mean()))
