"""Participant-held-out evaluation and held-out block permutation importance.

One invocation evaluates one fixed model/seed/video-label assignment over all folds.
Permutation 0 is observed; positive values permute the video-to-condition assignment
identically for all participants, then refit every fold. Existing result files are
resumed only if their complete configuration fingerprint agrees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from linear import DualShrinkageLDA

from data import (load_dataset, make_participant_folds, fit_fold_scaler,
                  model_data, ranking_groups, video_block_permutation,
                  permuted_video_labels, original_video_to_class, EMOTIONS)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from sdgnn import train_predict, IU

PRIMARY_SEED = 20261024
MODEL_NAMES = ['hybrid_gnn', 'sfgnn', 'lda_full']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def balanced_accuracy(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(len(EMOTIONS))]))


def unique_features(d):
    return np.concatenate([d['X'].reshape(len(d['y']), -1),
                           d['A'][:, :, IU[0], IU[1]].reshape(len(d['y']), -1)], axis=1)


def feature_columns(group, n_node_features=10):
    columns = []
    for j in group['node_indices']:
        columns.extend(10 * np.arange(30) + j)
    for j in group['edge_indices']:
        columns.extend(300 + 435 * j + np.arange(435))
    return np.array(columns, dtype=int)


def neural_probability(model, X, A, batch=256):
    results = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch):
            _, logits = model(torch.from_numpy(np.ascontiguousarray(X[start:start+batch])),
                              torch.from_numpy(np.ascontiguousarray(A[start:start+batch])))
            results.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(results)


def rank_fold(model_name, fitted, d, raw, te, groups, repeat_seeds,
              baseline_prob, baseline_features=None):
    """Perturb whole blocks in both graph and summary branches, never graph gates alone."""
    result = np.empty((len(groups), len(repeat_seeds), len(te)), dtype=np.uint8)
    if model_name == 'lda_full':
        base = fitted.decision_function(baseline_features[te])
    else:
        X, A = d['X'][te].copy(), d['A'][te].copy()
    for rep, seed in enumerate(repeat_seeds):
        donors = video_block_permutation(raw['subject'], raw['video'], te, int(seed))
        for gi, group in enumerate(groups):
            if model_name == 'lda_full':
                cols = feature_columns(group)
                delta = baseline_features[np.ix_(donors, cols)] - baseline_features[np.ix_(te, cols)]
                scores = base + delta @ fitted.coef_[:, cols].T
                result[gi, rep] = fitted.classes_[scores.argmax(axis=1)]
            else:
                for j in group['node_indices']:
                    X[:, :, j] = d['X'][donors, :, j]
                for j in group['edge_indices']:
                    A[:, j] = d['A'][donors, j]
                result[gi, rep] = neural_probability(fitted, X, A).argmax(axis=1)
                # Restore the baseline before modifying a different input group.
                for j in group['node_indices']:
                    X[:, :, j] = d['X'][te, :, j]
                for j in group['edge_indices']:
                    A[:, j] = d['A'][te, j]
    return result


def make_config(data, args, groups):
    return {
        'dataset': 'SEED-VII', 'n_classes': len(EMOTIONS),
        'protocol': 'Five-fold participant-held-out SEED-VII evaluation; familiar clips and sessions in all folds',
        'n_folds': args.folds, 'fold_seed': PRIMARY_SEED,
        'primary_seed': PRIMARY_SEED, 'sensitivity_seeds': [11, 29],
        'model_names': MODEL_NAMES, 'expected_permutations': 99,
        'epochs': args.epochs, 'feature_count': 5520,
        'standardization': 'Per scalar feature, training participants only; constant training features set to zero',
        'null': 'Permutation of 56 source-by-condition block labels shared by every participant; full refitting; preserves block-class counts, while clip-class counts may vary',
        'null_unit': 'source_condition_block',
        'session_scope': 'Familiar sessions; source-block null does not preserve session-by-condition counts; separate session-only control',
        'null_seed_offset': 10000, 'importance_repeats': args.importance_repeats,
        'importance': 'Whole input-block permutations across videos within each held-out participant; identical video reordering across held-out participants',
        'groups': groups, 'data_manifest': data['manifest'],
        'class_names': data['manifest']['emotions'],
        'source_sha256': {name: digest(HERE / name) for name in ['data.py', 'run.py', 'linear.py', 'vendor/sdgnn.py']},
        'selection': 'Fixed inherited architectures and hyperparameters; no outcome-based feature selection or tuning; exploratory extension',
        'training': {'hybrid_gnn': {'architecture': 'hybrid', 'lambda_video_adversary': 1.0},
                     'sfgnn': {'architecture': 'sfgnn', 'lambda_video_adversary': 0.0},
                     'lda_full': {'estimator': 'sklearn lsqr/auto equivalent',
                                  'implementation': 'float64 exact sample-space covariance solve',
                                  'shrinkage': 'auto'}},
        'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__},
    }


def run(args):
    started = time.time()
    torch.set_num_threads(args.threads)
    out = args.out_dir.resolve()
    for folder in ['runs', 'predictions', 'importance', 'models', 'progress']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.cache)
    groups = ranking_groups(data)
    cfg = make_config(data, args, groups)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    cfg_path = out / 'config.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise ValueError('Configuration differs from existing results; use a new output directory.')
    else:
        atomic_json(cfg_path, cfg)
    stem = f'{args.model}_seed{args.seed}_perm{args.perm:03d}'
    done = out / 'runs' / f'{stem}.json'
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['config_sha256'] != cfg_hash:
            raise ValueError('Saved task configuration mismatch')
        if not (out / previous['predictions_path']).exists():
            raise ValueError('Missing saved predictions')
        if args.importance and previous.get('importance_path') is None:
            raise ValueError('Task exists without importance; choose a separate output directory for rerun')
        print(json.dumps({'already_complete': str(done), 'balanced_accuracy': previous['balanced_accuracy']}), flush=True)
        return
    if args.perm:
        y, v2c = permuted_video_labels(data, 10000 + args.perm)
    else:
        y, v2c = data['y'].copy(), original_video_to_class(data)
    assert np.array_equal(y, v2c[data['video']])
    folds = make_participant_folds(data, n_splits=args.folds, seed=PRIMARY_SEED)
    probs = np.full((len(y), len(EMOTIONS)), np.nan, dtype=np.float32)
    fold_id = np.full(len(y), -1, dtype=int)
    repeat_seeds = np.arange(args.importance_repeats, dtype=np.int64) + 310000
    rank_pred = np.empty((len(groups), len(repeat_seeds), len(y)), dtype=np.uint8) if args.importance else None
    metrics = []
    for fold, (tr, te) in enumerate(folds):
        fold_started = time.time()
        scaler = fit_fold_scaler(data['X'], data['E'], tr)
        d = model_data(data, scaler)
        F = None
        if args.model == 'lda_full':
            F = unique_features(d).astype(np.float64)
            fitted = DualShrinkageLDA().fit(F[tr], y[tr])
            p = fitted.predict_proba(F[te])
        else:
            kind, lam = ('hybrid', 1.0) if args.model == 'hybrid_gnn' else ('sfgnn', 0.0)
            _, fitted = train_predict(d, tr, te, v2c, model=kind, lam=lam,
                                      epochs=args.epochs, seed=args.seed + fold,
                                      return_model=True)
            p = neural_probability(fitted, d['X'][te], d['A'][te])
        assert p.shape == (len(te), len(EMOTIONS)) and np.isfinite(p).all()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-5)
        probs[te] = p; fold_id[te] = fold
        if args.importance:
            rank_pred[:, :, te] = rank_fold(args.model, fitted, d, data, te, groups,
                                           repeat_seeds, p, baseline_features=F)
        if args.perm == 0:
            checkpoint = out / 'models' / f'{stem}_fold{fold}'
            # Scaler dictionaries may contain arrays; NPZ retains exact training statistics.
            np.savez_compressed(str(checkpoint) + '_scaler.npz', **scaler.to_dict(), train_idx=tr, test_idx=te)
            if args.model == 'lda_full':
                np.savez_compressed(str(checkpoint) + '_linear.npz', coef=fitted.coef_,
                                    intercept=fitted.intercept_, classes=fitted.classes_)
            else:
                torch.save({'state_dict': fitted.state_dict(), 'model': args.model,
                            'n_views': 12, 'n_features': 10, 'seed': args.seed + fold},
                           str(checkpoint) + '.pt')
        metrics.append({'fold': fold, 'balanced_accuracy': balanced_accuracy(y[te], p.argmax(axis=1)),
                        'train_subjects': sorted(np.unique(data['subject'][tr]).tolist()),
                        'test_subjects': sorted(np.unique(data['subject'][te]).tolist()),
                        'train_n': len(tr), 'test_n': len(te),
                        'seconds': round(time.time()-fold_started, 2)})
        atomic_json(out / 'progress' / f'{stem}.json', {'completed_folds': metrics, 'complete': False})
        print(json.dumps({'model': args.model, 'seed': args.seed, 'perm': args.perm, **metrics[-1]}), flush=True)
        del fitted, d, F
    assert (fold_id >= 0).all() and np.isfinite(probs).all()
    pred = probs.argmax(axis=1)
    predictions_path = Path('predictions') / f'{stem}.npz'
    np.savez_compressed(out / predictions_path, y=y, pred=pred, probabilities=probs,
                        subject=data['subject'], video=data['video'], idx=data['idx'], fold=fold_id,
                        session=data['session'], source_condition_block=data['source_condition_block'])
    importance_path = None
    if rank_pred is not None:
        importance_path = Path('importance') / f'{stem}.npz'
        np.savez_compressed(out / importance_path, predictions=rank_pred,
                            group_names=np.array([g['name'] for g in groups]),
                            group_kinds=np.array([g['kind'] for g in groups]),
                            repeat_seeds=repeat_seeds, baseline_predictions=pred)
    result = {'model': args.model, 'seed': args.seed, 'perm': args.perm,
              'balanced_accuracy': balanced_accuracy(y, pred), 'seconds': round(time.time()-started, 2),
              'config_sha256': cfg_hash, 'n_features': 5520,
              'predictions_path': str(predictions_path),
              'importance_path': str(importance_path) if importance_path else None,
              'fold_metrics': metrics}
    atomic_json(done, result)
    atomic_json(out / 'progress' / f'{stem}.json', {'completed_folds': metrics, 'complete': True})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=HERE.parent/'seed7_replication/cache')
    parser.add_argument('--out-dir', type=Path, default=HERE/'results')
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--seed', type=int, default=PRIMARY_SEED)
    parser.add_argument('--perm', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--importance-repeats', type=int, default=30)
    parser.add_argument('--importance', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.perm < 0 or (args.perm and args.importance):
        parser.error('Permutation id must be nonnegative; importance is for observed-label tasks only.')
    with threadpool_limits(limits=args.threads):
        run(args)
