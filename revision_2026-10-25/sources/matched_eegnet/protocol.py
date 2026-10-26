"""Freeze this benchmark before any production training or test-set scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import HERE, PRIMARY_SEED, SEEDS, MODELS, DATASETS, atomic_json, digest
from features import CONFIG as FEATURE_CONFIG


def initialize(prepared, out):
    sources = {p.name: digest(p) for p in sorted(HERE.glob('*.py'))}
    inputs = {}
    for dataset in DATASETS:
        inputs[dataset] = {}
        for seconds in [10, 30]:
            directory = prepared / dataset / f'{seconds}s'
            manifest = json.loads((directory / 'MANIFEST.json').read_text())
            for name, expected in manifest['output_sha256'].items():
                if digest(directory / name) != expected:
                    raise ValueError('Prepared input changed: ' + str(directory / name))
            with np.load(directory / 'meta.npz', allow_pickle=False) as meta:
                if len(np.unique(meta['subject'])) != DATASETS[dataset]['n_subjects']:
                    raise ValueError('Wrong participant count')
            inputs[dataset][str(seconds)] = manifest
    # Primary seed is scheduled first. Sensitivity seeds are never selected using outcomes.
    tasks = [dict(dataset=d, seconds=s, model=m, seed=seed, fold=f)
             for seed in SEEDS for d in DATASETS for s in [10,30] for m in MODELS for f in range(5)]
    config = dict(protocol_version=1, primary_seed=PRIMARY_SEED, seeds=SEEDS,
                  n_folds=5, datasets=DATASETS, seconds=[10,30], models=MODELS,
                  tasks=tasks, source_sha256=sources, prepared_manifest=inputs,
                  preprocessing=FEATURE_CONFIG,
                  training=dict(max_epochs=100, min_epochs=20, patience=20,
                                learning_rate=.001, batch_size=64),
                  evaluation=dict(primary_unit='30-second participant-video trial',
                                  ten_second_aggregation='mean of probabilities from three nonoverlapping windows',
                                  generalization='unseen participants viewing familiar video stimuli',
                                  bootstrap=2000, paired_sign_flips=9999,
                                  multiplicity='Holm across all 16 primary augmentation contrasts',
                                  inference='Approximate paired participant inference conditional on fitted OOF models; not a label-permutation test',
                                  duration_comparison='descriptive paired intervals; both use all 30 seconds',
                                  seeds='primary 20261024; sensitivity 11 and 29 reported separately'),
                  fusion=dict(connectivity_dim=4785, recurrence_dim=585, hidden_dim=32,
                              rule='add side-branch logits to the unchanged EEGNet classifier',
                              shuffled_control='joint side-vector trial permutation independently within participant; window positions preserved'),
                  architecture=dict(name='EEGNet-8,2', sampling_hz=200, temporal_kernel=100,
                                    separable_kernel=25, dropout=.5,
                                    source='https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py'))
    path = out / 'config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Refusing to replace an existing, different experiment configuration')
    else:
        atomic_json(path, config)
    print('Frozen configuration:', path, 'fold tasks:', len(tasks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    initialize(a.prepared, a.out)
