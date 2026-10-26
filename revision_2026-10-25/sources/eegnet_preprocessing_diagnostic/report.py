"""Descriptive audit summary; all recipes/seeds are shown, without winner selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from protocol import digest, read_config, RECIPES, SEEDS, save_json, metrics


def participant_scores(probability, y, subject):
    return np.array([metrics(probability[subject==s],y[subject==s])['balanced_accuracy']
                     for s in np.unique(subject)])


def interval(values):
    rng=np.random.default_rng(20261024)
    samples=rng.integers(0,len(values),size=(2000,len(values)))
    return np.quantile(values[samples].mean(1),[.025,.975]).tolist()


def trial_score(probability, expected):
    trials,first,inverse=np.unique(expected['trial_idx'],return_index=True,return_inverse=True)
    if not np.all(np.bincount(inverse)==3): raise ValueError('Incomplete trial windows')
    if not np.array_equal(expected['y'],expected['y'][first][inverse]):
        raise ValueError('Inconsistent trial labels')
    totals=np.zeros((len(trials),9)); np.add.at(totals,inverse,probability)
    return metrics(totals/3,expected['y'][first])


def report(config_path):
    config_path=Path(config_path); config=read_config(config_path); base=config_path.parent
    config_hash=digest(config_path); rows=[]; pending=[]; scores={}
    if digest(base/'metadata.npz')!=config['input_sha256']['metadata']:
        raise ValueError('Original metadata hash mismatch')
    with np.load(base/'metadata.npz',allow_pickle=False) as metadata:
        selected=np.flatnonzero((metadata['subject']>=100)&(metadata['subject']<123))
        expected={k:metadata[k][selected] for k in metadata.files}
    if len(selected)!=1932 or not np.array_equal(np.unique(expected['subject']),np.arange(100,123)):
        raise ValueError('Frozen metadata has wrong test coverage')
    for task in config['tasks']:
        folder=base/'runs'/f"{task['recipe']}_seed{task['seed']}"
        path=folder/'result.json'
        if not path.exists(): pending.append(task); continue
        result=json.loads(path.read_text())
        if not result['complete'] or result.get('smoke') or result['config_sha256']!=config_hash:
            raise ValueError('Invalid completed run: '+str(path))
        if any(result[k]!=v for k,v in task.items()): raise ValueError('Wrong task identity')
        if 'test_predictions.npz' not in result['artifact_sha256']:
            raise ValueError('Missing original prediction checksum')
        for name,expected_hash in result['artifact_sha256'].items():
            if digest(folder/name)!=expected_hash: raise ValueError('Artifact checksum mismatch')
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as data:
            for field,values in expected.items():
                if field not in data or not np.array_equal(data[field],values):
                    raise ValueError('Prediction metadata differs from frozen input: '+field)
            score=participant_scores(data['probabilities'],data['y'],data['subject'])
            measured=metrics(data['probabilities'],data['y'])
            trial_measured=trial_score(data['probabilities'],expected)
            if abs(measured['balanced_accuracy']-result['window_test']['balanced_accuracy'])>1e-12:
                raise ValueError('Saved balanced accuracy mismatch')
            if abs(trial_measured['balanced_accuracy']-result['trial_test']['balanced_accuracy'])>1e-12:
                raise ValueError('Saved trial balanced accuracy mismatch')
        scores[(task['recipe'],task['seed'])]=score
        low,high=interval(score)
        rows.append(dict(task,window_balanced_accuracy=measured['balanced_accuracy'],
                         participant_bootstrap_low=low,participant_bootstrap_high=high,
                         window_accuracy=measured['accuracy'],
                         trial_balanced_accuracy=trial_measured['balanced_accuracy'],
                         validation_balanced_accuracy=result['validation']['balanced_accuracy'],
                         selected_epoch=result['selected_epoch'],selection_epochs=result['selection_epochs']))
    differences=[]
    if not pending:
        for seed in SEEDS:
            for a,b in [(RECIPES[1],RECIPES[0]),(RECIPES[2],RECIPES[0]),(RECIPES[2],RECIPES[1])]:
                delta=scores[(a,seed)]-scores[(b,seed)]; low,high=interval(delta)
                differences.append(dict(seed=seed,recipe=a,reference=b,
                                        difference=float(delta.mean()),low=low,high=high))
    summary=dict(config_sha256=config_hash,complete=not pending,pending=pending,
                 results=rows,paired_differences=differences,
                 scope='Exploratory post-result processing diagnostic. Confidence intervals resample the 23 held-out participants conditional on each fitted model; they do not include training uncertainty. No p-values or test-based recipe selection.',
                 units='Accuracies and differences are proportions; multiply differences by 100 for percentage points.')
    save_json(base/'audit_summary.json',summary)
    if rows:
        with (base/'audit_scores.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    report(parser.parse_args().config)
