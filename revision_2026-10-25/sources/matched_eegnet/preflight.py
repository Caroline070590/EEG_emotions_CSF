"""Verify the deployed source and exact raw inputs before computing descriptors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from common import HERE, digest, DATASETS


def main(inputs):
    deployment = json.loads((HERE / 'deployment' / 'source_manifest.json').read_text())
    for name, expected in deployment.items():
        if digest(HERE / name) != expected:
            raise ValueError('Deployed source changed: ' + name)
    for dataset, item in json.loads(inputs.read_text()).items():
        for key in ['raw', 'metadata']:
            expected = item['source_sha256']['metadata_npz' if key == 'metadata' else key]
            if digest(item[key]) != expected:
                raise ValueError(dataset + ' input checksum failed: ' + key)
        signal = np.load(item['raw'], mmap_mode='r', allow_pickle=False)
        with np.load(item['metadata'], allow_pickle=False) as meta:
            n = DATASETS[dataset]['n_subjects'] * DATASETS[dataset]['n_videos']
            assert signal.shape == (n, 30, 7500)
            assert signal.dtype == np.float32
            assert len(meta['subject']) == n
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                    '-p', 'test_*.py', '-v'], check=True)
    print('Source, raw-input provenance and software tests passed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    main(p.parse_args().inputs)
