"""Build immutable-order trial metadata and remote input paths from local caches.

``build_inputs(out, remote_root, project_root, local_root=...)`` writes two
metadata NPZ files and inputs.json. All NPZ arrays are one value per raw trial;
there are no dataset-level arrays that would be repeated as trials by prepare.py.
Source EEG and CSV files are opened read-only and never copied or modified here.
"""
import argparse
import os
from pathlib import Path, PurePosixPath
import uuid

import numpy as np
import pandas as pd

from common import DATASETS, atomic_json, digest


LOCAL_ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    'faced': ('revision_2026-10/cache/eeg_30s.npy',
              'revision_2026-10/cache/features_scalar.csv'),
    'seed7': ('seed7_replication/cache/eeg_30s.npy',
              'seed7_replication/cache/meta.csv'),
}


def _integer_column(frame, name):
    if name not in frame:
        raise ValueError('Missing metadata column: ' + name)
    values = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Noninteger or missing metadata: ' + name)
    return values.astype(np.int64)


def validate_metadata(dataset, frame, raw_shape, raw_dtype, clips=None):
    """Return (per-trial NPZ arrays, JSON summary); reject changes in cache order.

    These two source caches are explicitly subject-major and video-minor. idx,
    subject and video must all agree with that order, without sorting any rows.
    SEED's independent clip table, when supplied, must agree on every clip field.
    """
    spec = DATASETS[dataset]
    ns, nv = spec['n_subjects'], spec['n_videos']
    n = ns * nv
    if tuple(raw_shape) != (n, 30, 7500) or len(frame) != n:
        raise ValueError('Raw shape and complete trial metadata do not match')
    if np.dtype(raw_dtype) != np.dtype('float32'):
        raise ValueError('Expected the original float32 EEG cache')
    values = {name: _integer_column(frame, name) for name in ['idx', 'subject', 'video']}
    if not np.array_equal(values['idx'], np.arange(n)):
        raise ValueError('idx must preserve the exact raw trial row order')
    offset = 0 if dataset == 'faced' else 1
    expected_subject = np.repeat(np.arange(offset, offset + ns), nv)
    expected_video = np.tile(np.arange(nv), ns)
    if not np.array_equal(values['subject'], expected_subject):
        raise ValueError('Participant order differs from the source cache')
    if not np.array_equal(values['video'], expected_video):
        raise ValueError('Video order differs from the source cache')
    if len(np.unique(np.c_[values['subject'], values['video']], axis=0)) != n:
        raise ValueError('Duplicate participant-video trials')
    if 'emotion' not in frame or frame['emotion'].isna().any():
        raise ValueError('Missing emotion labels')
    emotions = frame['emotion'].to_numpy(dtype=str)
    labels = spec['class_names']
    if set(emotions) != set(labels):
        raise ValueError('Emotion labels differ from common.DATASETS')
    y = np.array([labels.index(x) for x in emotions], dtype=np.int64)
    video_y = y[:nv]
    if not np.array_equal(y, np.tile(video_y, ns)):
        raise ValueError('The emotion for a video differs between participants')
    expected_counts = [4 if x == 'neutral' else 3 for x in labels] if dataset == 'faced' else [
        8 if x == 'neutral' else 12 for x in labels]
    counts = np.bincount(video_y, minlength=len(labels))
    if not np.array_equal(counts, expected_counts):
        raise ValueError('Emotion counts do not match the complete clip design')
    if dataset == 'faced':
        if not np.array_equal(video_y, np.repeat(np.arange(len(labels)), expected_counts)):
            raise ValueError('FACED emotion/video mapping differs from the source cache')
    values['y'] = y
    summary = {
        'shape': list(raw_shape), 'dtype': str(np.dtype(raw_dtype)),
        'sampling_hz': 250, 'duration_seconds': 30,
        'n_trials': n, 'n_subjects': ns, 'n_videos': nv,
        'subject_ids': list(range(offset, offset + ns)),
        'class_names': list(labels), 'class_counts': (counts * ns).tolist(),
        'video_class_counts': counts.tolist(), 'video_to_class': video_y.tolist(),
        'trial_order': 'source row order, subject-major and video-minor; idx equals raw row',
    }
    if dataset == 'seed7':
        values['clip'] = _integer_column(frame, 'clip')
        values['session'] = _integer_column(frame, 'session')
        if not np.array_equal(values['clip'], values['video'] + 1):
            raise ValueError('SEED clip IDs must be video IDs plus one')
        if not np.array_equal(values['session'], np.tile(np.repeat(np.arange(1, 5), 20), ns)):
            raise ValueError('SEED session IDs differ from the source design')
        for name in ['source', 'title']:
            if name not in frame or frame[name].isna().any() or (frame[name].str.len() == 0).any():
                raise ValueError('Missing SEED ' + name)
            if not np.array_equal(frame[name].to_numpy(), np.tile(frame[name].to_numpy()[:nv], ns)):
                raise ValueError('Inconsistent SEED clip metadata: ' + name)
        if clips is not None:
            if len(clips) != nv or not np.array_equal(_integer_column(clips, 'clip'), np.arange(1, nv + 1)):
                raise ValueError('Independent clip table is incomplete or out of order')
            for name in ['clip', 'session', 'emotion', 'source', 'title']:
                if name not in clips or not np.array_equal(frame[name].to_numpy(), np.tile(clips[name].to_numpy(), ns)):
                    raise ValueError('SEED metadata disagrees with clip table: ' + name)
        block_strings = np.char.add(np.char.add(frame['source'].to_numpy(dtype=str), '|'), emotions)
        block_names, source_block = np.unique(block_strings, return_inverse=True)
        values['source_block'] = source_block.astype(np.int64)
        video_blocks = source_block[:nv]
        block_labels = [int(y[np.flatnonzero(source_block == b)[0]]) for b in range(len(block_names))]
        summary.update(n_sources=int(frame['source'].nunique()),
                       n_source_condition_blocks=len(block_names),
                       source_block_names=block_names.tolist(),
                       block_labels=block_labels, video_to_block=video_blocks.tolist(),
                       video_to_session=values['session'][:nv].tolist(),
                       source_block_definition='source programme plus original emotion condition')
    summary['metadata_fields'] = list(values)
    return values, summary


def _remote_path(value):
    path = PurePosixPath(str(value))
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Remote roots must be absolute POSIX paths without parent traversal')
    return path


def build_inputs(out, remote_root, project_root, local_root=LOCAL_ROOT):
    """Validate sources, write trial NPZs and return the inputs.json dictionary."""
    out, local_root = Path(out), Path(local_root)
    remote_root, project_root = _remote_path(remote_root), _remote_path(project_root)
    # Validate both datasets before creating any deliverable.
    validated = {}
    for dataset, (raw_rel, csv_rel) in SOURCES.items():
        raw_path, csv_path = local_root / raw_rel, local_root / csv_rel
        raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
        columns = ['idx', 'subject', 'video', 'emotion']
        if dataset == 'seed7':
            columns += ['clip', 'session', 'source', 'title']
        frame = pd.read_csv(csv_path, usecols=columns)
        clips_path = csv_path.with_name('clips.csv') if dataset == 'seed7' else None
        clips = pd.read_csv(clips_path) if clips_path is not None else None
        arrays, summary = validate_metadata(dataset, frame, raw.shape, raw.dtype, clips)
        del raw
        validated[dataset] = (arrays, summary, raw_path, csv_path, clips_path)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for dataset, (arrays, summary, raw_path, csv_path, clips_path) in validated.items():
        metadata_path = out / (dataset + '_meta.npz')
        temporary = metadata_path.with_name(metadata_path.stem + '.' + uuid.uuid4().hex + '.npz')
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, metadata_path)
        raw_remote = project_root / 'cache/eeg_30s.npy' if dataset == 'faced' else remote_root / 'input/seed7_eeg_30s.npy'
        hashes = {'raw': digest(raw_path), 'metadata_csv': digest(csv_path),
                  'metadata_npz': digest(metadata_path)}
        source_files = {'raw': str(raw_path.resolve()), 'metadata_csv': str(csv_path.resolve())}
        if clips_path is not None:
            hashes['clips_csv'] = digest(clips_path)
            source_files['clips_csv'] = str(clips_path.resolve())
        inputs[dataset] = dict(summary, raw=str(raw_remote),
                               metadata=str(remote_root / 'input' / metadata_path.name),
                               source_sha256=hashes, local_source_files=source_files)
        print(f'{dataset}: validated {summary["n_trials"]} trials; raw SHA256 {hashes["raw"]}', flush=True)
    atomic_json(out / 'inputs.json', inputs)
    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--local-root', type=Path, default=LOCAL_ROOT)
    args = parser.parse_args()
    build_inputs(args.out, args.remote_root, args.project_root, args.local_root)
