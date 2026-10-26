"""Fixed protocol and reproducible identities for the matched EEGNet experiment."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
PRIMARY_SEED = 20261024
SEEDS = [PRIMARY_SEED, 11, 29]
MODELS = ['eegnet', 'eegnet_connectivity', 'eegnet_recurrence',
          'eegnet_combined', 'eegnet_shuffled']
DATASETS = {
    'faced': {'n_subjects': 123, 'n_videos': 28, 'class_names':
              ['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']},
    'seed7': {'n_subjects': 20, 'n_videos': 80, 'class_names':
              ['anger','disgust','fear','sadness','neutral','happiness','surprise']},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def participant_split(subject, fold, seed=PRIMARY_SEED):
    """Same five outer folds as the existing evaluations; inner split never uses test."""
    subject = np.asarray(subject)
    people = np.unique(subject)
    if len(people) < 10 or fold not in range(5):
        raise ValueError('Five folds and at least ten participants are required')
    held = np.array_split(np.random.default_rng(seed).permutation(people), 5)[fold]
    outer = np.setdiff1d(people, held)
    shuffled = np.random.default_rng(seed + 1000 + fold).permutation(outer)
    n_valid = max(2, int(np.ceil(.2 * len(outer))))
    valid = shuffled[:n_valid]; train = shuffled[n_valid:]
    return {key: np.flatnonzero(np.isin(subject, value)) for key, value in
            [('outer_train',outer), ('inner_train',train), ('validation',valid), ('test',held)]}


def trial_probabilities(probabilities, trial_idx, y):
    trials, first, inverse = np.unique(trial_idx, return_index=True, return_inverse=True)
    if not np.array_equal(y, y[first][inverse]):
        raise ValueError('Windows of a trial have different labels')
    summed = np.zeros((len(trials), probabilities.shape[1]), dtype=np.float64)
    np.add.at(summed, inverse, probabilities)
    return summed / np.bincount(inverse)[:, None], y[first], trials


def balanced_accuracy(y, probabilities):
    pred = probabilities.argmax(1)
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(probabilities.shape[1])]))


def side_donors(meta, indices, seed):
    """Independent participant-wise trial shuffles; all windows of a donor stay together."""
    indices = np.asarray(indices)
    donors = np.full(len(meta['subject']), -1, dtype=np.int64)
    for person in np.unique(meta['subject'][indices]):
        rows = indices[meta['subject'][indices] == person]
        trials = np.unique(meta['trial_idx'][rows])
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(person)]))
        mapping = dict(zip(trials.tolist(), rng.permutation(trials).tolist()))
        lookup = {(int(meta['trial_idx'][i]), int(meta['window_id'][i])): int(i) for i in rows}
        for i in rows:
            donors[i] = lookup[(mapping[int(meta['trial_idx'][i])], int(meta['window_id'][i]))]
    if np.any(donors[indices] < 0) or not np.array_equal(np.sort(donors[indices]), np.sort(indices)):
        raise ValueError('Invalid within-participant donor permutation')
    return donors
