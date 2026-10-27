from pathlib import Path
import sys,json
import mne
import numpy as np
from scipy.spatial.distance import cdist
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'core'))
from analysis_core import CHANNELS
pos={k.upper():v for k,v in mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos'].items()}
xyz=np.array([pos[c] for c in CHANNELS]);d=cdist(xyz,xyz);scale=np.median(d[np.triu_indices(30,1)])
a=np.exp(-(d/scale)**2);np.fill_diagonal(a,0)
np.savez(HERE/'geometry.npz',adjacency=a.astype('float32'),coordinates=xyz,channels=CHANNELS)
(HERE/'geometry.json').write_text(json.dumps({'source':'MNE standard_1020','mne_version':mne.__version__,'coordinates':'template positions in metres; not measured individual electrode locations','kernel':'exp(-(Euclidean distance / median off-diagonal distance)^2); diagonal zero','scale_metres':scale,'channels':CHANNELS},indent=2))
