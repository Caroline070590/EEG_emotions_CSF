"""Compute features independently for each 10-s or 30-s EEG input window."""
import argparse
import json
import os
from pathlib import Path
import uuid
import numpy as np
from common import DATASETS, atomic_json, digest, fingerprint
from features import process_window


def chunk(args):
    inputs = json.loads(args.inputs.read_text())
    item = inputs[args.dataset]
    trials = np.load(item['metadata'], allow_pickle=False)
    raw = np.load(item['raw'], mmap_mode='r', allow_pickle=False)
    if raw.shape != (len(trials['subject']), 30, 7500):
        raise ValueError('Expected complete 30-s input trials in metadata order')
    directory = args.prepared / args.dataset / f'{args.seconds}s'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'chunks' / f'chunk_{args.chunk:03d}.npz'
    path.parent.mkdir(exist_ok=True)
    cfg_hash = fingerprint({'input': item, 'seconds': args.seconds, 'chunks': args.chunks,
                            'features_sha256': digest(Path(__file__).with_name('features.py')),
                            'prepare_sha256': digest(Path(__file__))})
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['config_sha256']) != cfg_hash:
                raise ValueError('Existing feature chunk has different provenance')
        print('Already complete', path, flush=True); return
    selected = np.array_split(np.arange(len(raw)), args.chunks)[args.chunk]
    signals, sides, ids = [], [], []
    per_trial = 30 // args.seconds
    for trial in selected:
        for w in range(per_trial):
            start = w * args.seconds * 250
            values = process_window(raw[trial, :, start:start + args.seconds * 250])
            signals.append(values['signal'])
            sides.append(np.r_[values['connectivity'], values['recurrence']])
            ids.append(int(trial) * per_trial + w)
    temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.npz')
    np.savez_compressed(temporary, window_idx=np.array(ids), signal=np.stack(signals),
                        side=np.stack(sides), config_sha256=np.array(cfg_hash))
    os.replace(temporary, path)
    print(json.dumps({'dataset': args.dataset, 'seconds': args.seconds, 'chunk': args.chunk,
                      'windows': len(ids)}), flush=True)


def merge(args):
    inputs = json.loads(args.inputs.read_text()); item = inputs[args.dataset]
    with np.load(item['metadata'], allow_pickle=False) as saved:
        trials = {k: saved[k] for k in saved.files}
    directory = args.prepared / args.dataset / f'{args.seconds}s'
    per_trial = 30 // args.seconds; n = len(trials['subject']) * per_trial
    cfg_hash = fingerprint({'input': item, 'seconds': args.seconds, 'chunks': args.chunks,
                            'features_sha256': digest(Path(__file__).with_name('features.py')),
                            'prepare_sha256': digest(Path(__file__))})
    complete = directory / 'MANIFEST.json'
    if complete.exists():
        old = json.loads(complete.read_text())
        if old['chunk_config_sha256'] != cfg_hash:
            raise ValueError('Prepared cache configuration changed')
        for name, expected in old['output_sha256'].items():
            if digest(directory / name) != expected:
                raise ValueError('Prepared cache changed: ' + name)
        print('Already merged', directory, flush=True); return
    signal = np.lib.format.open_memmap(directory / 'signal.npy', mode='w+', dtype=np.float32,
                                       shape=(n, 30, args.seconds * 200))
    side = np.lib.format.open_memmap(directory / 'side.npy', mode='w+', dtype=np.float32, shape=(n, 5370))
    coverage = np.zeros(n, dtype=np.int64)
    for i in range(args.chunks):
        with np.load(directory / 'chunks' / f'chunk_{i:03d}.npz', allow_pickle=False) as part:
            if str(part['config_sha256']) != cfg_hash:
                raise ValueError('Chunk provenance mismatch')
            rows = part['window_idx']
            if len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= n:
                raise ValueError('Invalid chunk indices')
            if not np.isfinite(part['signal']).all() or not np.isfinite(part['side']).all():
                raise ValueError('Nonfinite prepared inputs')
            signal[rows] = part['signal']; side[rows] = part['side']; coverage[rows] += 1
    if not np.all(coverage == 1):
        raise ValueError('Window coverage is not exactly once')
    signal.flush(); side.flush(); del signal, side
    meta = {k: np.repeat(v, per_trial) for k, v in trials.items()}
    meta.update(trial_idx=np.repeat(np.arange(len(trials['subject'])), per_trial),
                window_idx=np.arange(n), window_id=np.tile(np.arange(per_trial), len(trials['subject'])),
                window_start=np.tile(np.arange(per_trial) * args.seconds, len(trials['subject'])))
    np.savez_compressed(directory / 'meta.npz', **meta)
    atomic_json(complete, {'dataset': args.dataset, 'seconds': args.seconds, 'n_windows': n,
                          'n_trials': len(trials['subject']), 'chunk_config_sha256': cfg_hash,
                          'input': item, 'side_features': {'connectivity': [0,4785], 'recurrence': [4785,5370]},
                          'output_sha256': {name: digest(directory/name) for name in ['signal.npy','side.npy','meta.npz']}})
    print('Merged', directory, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['chunk','merge'])
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--dataset', choices=list(DATASETS), required=True)
    p.add_argument('--seconds', type=int, choices=[10,30], required=True)
    p.add_argument('--chunks', type=int, default=64)
    p.add_argument('--chunk', type=int, default=0)
    args = p.parse_args()
    {'chunk': chunk, 'merge': merge}[args.stage](args)
