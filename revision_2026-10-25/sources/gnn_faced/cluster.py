"""Deploy the fixed participant protocol as independently resumable Slurm tasks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MODELS = ['hybrid_gnn', 'sfgnn', 'lda_full']
SEED = 20261024
TASKS = [(m, SEED, 0, True) for m in MODELS]
TASKS += [(m, SEED, p, False) for p in range(1, 100) for m in MODELS]
TASKS += [(m, s, 0, True) for s in [11, 29] for m in MODELS[:2]]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight(args):
    manifest = json.loads((HERE / 'deployment_manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if sha256(HERE / name) != expected:
            raise ValueError(f'Deployed source differs: {name}')
    for name, expected in manifest['input_sha256'].items():
        if sha256(args.cache / name) != expected:
            raise ValueError(f'Cluster input differs from the validated local input: {name}')
    subprocess.run([sys.executable, '-m', 'unittest', 'test_data', 'test_linear',
                    'test_runner', '-v'], cwd=HERE, check=True)
    # Initialize configuration once before the array. All tasks then only read it.
    import run
    from data import load_dataset, ranking_groups
    data = load_dataset(args.cache)
    cfg = run.make_config(data, SimpleNamespace(folds=5, epochs=40,
                                               importance_repeats=30), ranking_groups(data))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / 'config.json'
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise ValueError('Cluster configuration changed; use a new output directory')
    if not path.exists():
        run.atomic_json(path, cfg)
    validation = {
        'source_and_input_hashes_verified': True,
        'numeric_and_leakage_tests_passed': True,
        'python': sys.version,
        'packages': {p: importlib.metadata.version(p) for p in
                     ['numpy', 'pandas', 'torch', 'scipy', 'scikit-learn', 'threadpoolctl']},
        'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        'tasks': [{'index': i, 'model': m, 'seed': s, 'permutation': p, 'importance': imp}
                  for i, (m, s, p, imp) in enumerate(TASKS)],
    }
    run.atomic_json(args.out_dir / 'CLUSTER_PREFLIGHT.json', validation)
    print(json.dumps({'preflight': 'passed', 'tasks': len(TASKS)}), flush=True)


def task(args):
    index = args.index
    if index is None:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(TASKS):
        raise ValueError(f'Invalid task index {index}')
    m, seed, perm, importance = TASKS[index]
    cmd = [sys.executable, str(HERE / 'run.py'), '--model', m, '--seed', str(seed),
           '--perm', str(perm), '--threads', '1', '--cache', str(args.cache),
           '--out-dir', str(args.out_dir)]
    if importance:
        cmd.append('--importance')
    subprocess.run(cmd, cwd=HERE, check=True)


def report(args):
    for m, seed, perm, importance in TASKS:
        path = args.out_dir / 'runs' / f'{m}_seed{seed}_perm{perm:03d}.json'
        result = json.loads(path.read_text())
        if (result['model'], result['seed'], result['perm']) != (m, seed, perm):
            raise ValueError(f'Task identity mismatch: {path}')
        if not (args.out_dir / result['predictions_path']).is_file():
            raise ValueError(f'Missing predictions: {path}')
        if importance and not (args.out_dir / result['importance_path']).is_file():
            raise ValueError(f'Missing importance: {path}')
    subprocess.run([sys.executable, str(HERE / 'report.py'), '--out-dir', str(args.out_dir),
                    '--expected-permutations', '99', '--bootstrap', '2000'],
                   cwd=HERE, check=True)
    print(json.dumps({'completed_tasks': len(TASKS), 'report': 'complete'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'task', 'report'])
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=HERE / 'cluster_results')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    {'preflight': preflight, 'task': task, 'report': report}[args.stage](args)
