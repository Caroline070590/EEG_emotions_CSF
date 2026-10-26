"""Portable Figure1 release check: run actual package code in a temporary sandbox.

Use --release-root PACKAGE --out OUTPUT_DIRECTORY. The package is not modified.
The sibling src/cache and reference_rasters are small retained metric fixtures,
not raw EEG. No assertion about an old import failure is made here.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import argparse
import atexit
import shutil
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--release-root',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
release=args.release_root.resolve()
candidates=[release/'faced_1s_exploratory/src',release/'src',release]
source=next((x for x in candidates if (x/'part1_topology/run_part1_q30.py').is_file()),None)
if source is None: raise ValueError('Cannot resolve faced_1s_exploratory/src in release-root')
args.out.mkdir(parents=True,exist_ok=True)
args.out=args.out.resolve()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source_hashes={str(path.relative_to(source)):sha(path) for folder in ['lib','part1_topology'] for path in sorted((source/folder).rglob('*.py'))}
fixture_hashes={str(path.relative_to(HERE)):sha(path) for folder in ['src/cache','reference_rasters'] for path in sorted((HERE/folder).glob('*')) if path.is_file()}
sandbox=tempfile.TemporaryDirectory(prefix='figure1_release_')
atexit.register(sandbox.cleanup)
package=Path(sandbox.name)/'src'
for folder in ['lib','part1_topology']:
    shutil.copytree(source/folder,package/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(HERE/'src/cache',package/'cache')
sys.path.insert(0,str(package))
from part1_topology import run_part1_q30 as fixed
from part1_topology import run_part1 as helpers
from lib.config import EMOS, METRICS

checks = {}
for i, tag in enumerate(['spearman','mtf_q30','qg_q30']):
    d = fixed._load(tag)
    assert len(d)==3444
    z = helpers._profile(d).to_numpy()
    image = np.asarray(Image.open(HERE/'reference_rasters'/f'panel-{i:03d}.png').convert('RGB'))
    h,w = image.shape[:2]
    actual = np.array([[image[int((row+.5)*h/9),int((col+.5)*w/6)] for col in range(6)] for row in range(9)])
    expected = (matplotlib.colormaps['viridis'](Normalize(-2,2)(z))[:,:,:3]*255).astype(np.uint8)
    error = np.max(abs(actual.astype(int)-expected.astype(int)),axis=2)
    checks[tag] = dict(rows=len(d), finite_metric_counts=d[METRICS].notna().sum().to_dict(),
                      exact_fingerprint_cells=int((error==0).sum()), total_cells=54,
                      max_rgb_difference=int(error.max()))
    assert (error==0).all()

with tempfile.TemporaryDirectory() as temporary:
    tmp = Path(temporary)
    bad = fixed._load('mtf_q30').copy();bad['q']=13
    bad.to_csv(tmp/'trial_metrics_mtf.csv',index=False)
    with patch.object(fixed,'CACHE',tmp):
        try:fixed._load('mtf_q30')
        except ValueError:pass
        else:raise AssertionError('Historical q=13 cache was accepted')

# A plotting-helper check against independent vectorized participant bootstrap.
import matplotlib.pyplot as plt
d = fixed._load('spearman');metric='sigma_small_world'
participants = d.groupby(['subject','emotion'])[metric].mean().reset_index()
rng = np.random.default_rng(0);expected_means=[];expected_bounds=[]
for emotion in EMOS:
    values=participants.loc[participants.emotion.eq(emotion),metric].dropna().to_numpy()
    expected_means.append(values.mean())
    boot=rng.choice(values,size=(1200,len(values)),replace=True).mean(axis=1)
    expected_bounds.append(np.percentile(boot,[2.5,97.5]))
helpers.RNG=np.random.default_rng(0)
figure,axis=plt.subplots();helpers.metric_bars(axis,d,metric)
np.testing.assert_allclose([p.get_height() for p in axis.patches],expected_means,atol=1e-14)
segments=axis.containers[0].lines[2][0].get_segments()
np.testing.assert_allclose([[line[0,1],line[1,1]] for line in segments],expected_bounds,atol=1e-14)
plt.close(figure)
helpers.RNG=np.random.default_rng(0)
fixed.FIG_DIR=args.out/'figures'
fixed.main()
pdf=args.out/'figures/part1/Figure_Part1_global_topology_q30.pdf'
assert pdf.exists() and pdf.stat().st_size>1000
record=dict(status='passed',release_root=str(release),resolved_source=str(source),
            actual_release_source_sha256=source_hashes,fixture_sha256=fixture_hashes,
            imported_generator=True,q13_input_rejected=True,
            fingerprint_checks=checks,bootstrap_mean_and_intervals_verified=True,
            generated_pdf_sha256=sha(pdf),
            limitation='Checks the selected release generator against retained metric fixtures and original numerical fingerprints. It does not rerun raw topology or assert PDF-byte identity.')
assert all(sha(source/name)==expected for name,expected in source_hashes.items()),'Source modified during test'
(args.out/'RELEASE_FIGURE1_SMOKE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['status','resolved_source','q13_input_rejected','bootstrap_mean_and_intervals_verified']},indent=2))
