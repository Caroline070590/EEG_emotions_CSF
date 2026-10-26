"""Recording-session-only diagnostic, with every decision fitted on training participants.

This control uses no EEG. Its class-balanced decision maximizes P(session | class)
estimated from the training fold, with ties resolved by the fixed class order.
It evaluates a known recording-design nuisance, not emotion physiology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_predict_session(session, y, train, test, n_classes):
    session, y = np.asarray(session), np.asarray(y)
    train, test = np.asarray(train), np.asarray(test)
    counts = np.bincount(y[train], minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError('Every condition must occur in the training participants')
    pred = np.empty(len(test), dtype=int)
    for value in np.unique(session[test]):
        same = train[session[train] == value]
        if not len(same):
            raise ValueError('Test session is absent from training participants')
        conditional_session = np.bincount(y[same], minlength=n_classes) / counts
        pred[session[test] == value] = int(np.argmax(conditional_session))
    return pred


def main(args):
    from data import EMOTIONS, load_dataset, make_participant_folds, permuted_video_labels
    from report import participant_bootstrap, paired_model_difference
    from run import atomic_json

    data = load_dataset(args.cache)
    config = json.loads((args.out_dir / 'config.json').read_text())
    if config['data_manifest'] != data['manifest']:
        raise ValueError('Session control data differ from the model evaluation')
    n_classes = len(EMOTIONS)
    folds = make_participant_folds(data, n_splits=config['n_folds'], seed=config['fold_seed'])
    rows, observed = [], None
    for perm in range(config['expected_permutations'] + 1):
        y = data['y'] if perm == 0 else permuted_video_labels(data, 10000 + perm)[0]
        pred = np.full(len(y), -1, dtype=int)
        fold = np.full(len(y), -1, dtype=int)
        for i, (tr, te) in enumerate(folds):
            pred[te] = fit_predict_session(data['session'], y, tr, te, n_classes)
            fold[te] = i
        if np.any(pred < 0):
            raise ValueError('Incomplete participant coverage')
        score = float(np.mean([np.mean(pred[y == c] == c) for c in range(n_classes)]))
        rows.append({'permutation': perm, 'balanced_accuracy_percent': 100 * score})
        if perm == 0:
            observed = dict(y=y, pred=pred, subject=data['subject'], video=data['video'],
                            idx=data['idx'], fold=fold, session=data['session'])
    np.savez_compressed(args.out_dir / 'session_control.npz', **observed)
    pd.DataFrame(rows).to_csv(args.out_dir / 'session_control_scores.csv', index=False)
    uncertainty = participant_bootstrap(observed['y'], observed['pred'], observed['subject'],
                                        n_classes, 2000, config['primary_seed'])
    comparisons = []
    for model in config['model_names']:
        path = args.out_dir / 'predictions' / f'{model}_seed{config["primary_seed"]}_perm000.npz'
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            eeg = {k: saved[k] for k in ['y', 'pred', 'subject', 'video', 'idx', 'fold']}
        for key in ['y', 'subject', 'video', 'idx', 'fold']:
            np.testing.assert_array_equal(eeg[key], observed[key])
        comparisons.append({'eeg_model': model, 'control': 'session_only',
                            **paired_model_difference(eeg, observed, n_classes, 2000,
                                                      config['primary_seed'])})
    pd.DataFrame(comparisons).to_csv(args.out_dir / 'session_control_paired_differences.csv', index=False)
    result = {
        'dataset': 'SEED-VII',
        'definition': 'Train-fold balanced-accuracy-optimal prediction from session identity alone; deterministic class-order ties; no EEG input',
        'scope': 'Diagnostic for the condition imbalance across familiar recording sessions; not a conditional permutation test or evidence of emotion-specific information',
        'generalization': 'Unseen participants viewing familiar clips in familiar sessions',
        'bootstrap_scope': 'Participant resampling conditional on this fixed session/clip design; identical participant labels yield degenerate control-score intervals',
        'observed': uncertainty,
        'scores': rows,
        'paired_eeg_minus_control': comparisons,
        'null_scope': 'Same 99 source-by-condition label permutations as the EEG models; block-class counts preserved, session-class counts not preserved',
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(args.out_dir / 'session_control.json', result)
    print(json.dumps({'session_only_balanced_accuracy_percent': rows[0]['balanced_accuracy_percent'],
                      'paired_comparisons': len(comparisons)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    main(parser.parse_args())
