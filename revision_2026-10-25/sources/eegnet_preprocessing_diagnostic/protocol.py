"""Freeze a bounded, explicitly exploratory FACED EEGNet processing audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLASSES = ['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']
RECIPES = ['window_filtered_zscore', 'trial_fourier_div100', 'trial_fourier_zscore']
SIGNAL_NAMES = {'window_filtered_zscore':'window_filtered_zscore.npy',
                'trial_fourier_div100':'trial_fourier.npy',
                'trial_fourier_zscore':'trial_fourier.npy'}
SEEDS = [20261024, 11, 29]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sources():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.slurm')) + list((HERE/'provenance').glob('*'))
    return {str(p.relative_to(HERE)): digest(p) for p in sorted(paths) if p.is_file()}


def metrics(probabilities, y):
    probabilities = np.asarray(probabilities); y = np.asarray(y)
    if probabilities.shape != (len(y),9) or set(np.unique(y)) != set(range(9)):
        raise ValueError('Every evaluated set must contain all nine classes')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(1),1,atol=1e-5):
        raise ValueError('Invalid probability vectors')
    pred = probabilities.argmax(1)
    recalls = [float(np.mean(pred[y==c] == c)) for c in range(9)]
    return {'balanced_accuracy':float(np.mean(recalls)), 'accuracy':float(np.mean(pred==y)),
            'cross_entropy':float(-np.log(np.maximum(probabilities[np.arange(len(y)),y],1e-30)).mean()),
            'class_recalls':recalls,'predicted_class_counts':np.bincount(pred,minlength=9).tolist()}


def read_config(path):
    value = json.loads(Path(path).read_text())
    if value['source_sha256'] != sources():
        raise ValueError('Audit source differs from frozen protocol')
    if value['recipes'] != RECIPES or value['seeds'] != SEEDS:
        raise ValueError('Recipes or seeds differ from the bounded audit')
    expected = [dict(recipe=r, seed=s) for r in RECIPES for s in SEEDS]
    if value['tasks'] != expected:
        raise ValueError('Task list differs from nine prespecified fits')
    return value


def validate_trials(raw, frame):
    if raw.shape != (123 * 28, 30, 7500) or raw.dtype != np.float32:
        raise ValueError('This adaptation requires the verified 30-channel float32 FACED cache')
    expected = {'idx': np.arange(123 * 28),
                'subject': np.repeat(np.arange(123), 28),
                'video': np.tile(np.arange(28), 123)}
    for key, values in expected.items():
        if key not in frame or not np.array_equal(frame[key].to_numpy(), values):
            raise ValueError('FACED metadata/order mismatch: ' + key)
    video_y = np.repeat(np.arange(9), [3,3,3,3,4,3,3,3,3])
    y = np.tile(video_y, 123)
    if not np.array_equal(frame['emotion'].to_numpy(), np.asarray(CLASSES)[y]):
        raise ValueError('FACED emotion mapping mismatch')
    return dict(expected, y=y)


def window_metadata(trials):
    result = {k: np.repeat(v, 3) for k, v in trials.items() if k != 'idx'}
    result.update(trial_idx=np.repeat(trials['idx'], 3),
                  window_id=np.tile(np.arange(3), len(trials['idx'])),
                  window_idx=np.arange(len(trials['idx']) * 3))
    result['window_start'] = result['window_id'] * 10
    return result


def fixed_splits(subject):
    subject = np.asarray(subject)
    if set(np.unique(subject)) != set(range(123)):
        raise ValueError('Fixed split requires all 123 zero-based FACED participants')
    return {'train': np.flatnonzero(subject < 80),
            'validation': np.flatnonzero((subject >= 80) & (subject < 100)),
            'test': np.flatnonzero(subject >= 100)}


def create(raw_path, csv_path, out):
    out = Path(out)
    if (out / 'config.json').exists():
        raise FileExistsError('Refusing to overwrite a frozen audit protocol')
    raw_path, csv_path = Path(raw_path).resolve(), Path(csv_path).resolve()
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    frame = pd.read_csv(csv_path, usecols=['idx','subject','video','emotion'])
    trials = validate_trials(raw, frame)
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / 'metadata.npz'
    np.savez_compressed(metadata_path, **window_metadata(trials))
    config = dict(
        schema_version=1, dataset='FACED', exploratory=True,
        description='Post-result diagnostic of three prespecified processing recipes; not an exact published EEGNet replication',
        n_channels=30, n_classes=9, class_names=CLASSES,
        raw_path=str(raw_path), metadata_csv=str(csv_path),
        input_sha256={'raw':digest(raw_path),'metadata_csv':digest(csv_path),
                      'metadata':digest(metadata_path)},
        source_sha256=sources(), recipes=RECIPES, seeds=SEEDS,
        tasks=[dict(recipe=r, seed=s) for r in RECIPES for s in SEEDS],
        participants={'train':list(range(80)), 'validation':list(range(80,100)), 'test':list(range(100,123))},
        training={'max_epochs':100, 'min_epochs':20, 'patience':20, 'batch_size':64,
                  'optimizer':'Adam', 'learning_rate':0.001, 'weight_decay':0,
                  'loss':'training-class-frequency-weighted cross-entropy',
                  'selection':'validation window balanced accuracy; lower unweighted window cross-entropy breaks ties',
                  'refit':False, 'scheduler':None},
        primary_metric='10-second window balanced accuracy',
        secondary_metric='30-second trial balanced accuracy from mean of three window probabilities',
        interpretation='Familiar videos in held-out participants; same participants have been examined in earlier experiments. No independent confirmatory test, new-stimulus claim, or exact reproduction claim.',
        comparison='B versus C isolates scaling; A versus C compares the filtering/resampling pipeline at the same normalization. A versus B bundles these changes. Resampling and filtering are not separated.',
        units='Original cache amplitudes retained; physical units not independently verified. Division by 100 follows the released FACED loader literally.',
        test_policy='Once per completed recipe/seed after validation-only checkpoint selection. All nine prespecified results must be reported; no recipe chosen using test scores.')
    save_json(out / 'config.json', config)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--metadata-csv', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    create(args.raw, args.metadata_csv, args.out)
