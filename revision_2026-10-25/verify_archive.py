"""Verify bundled run/config/artifact identity and independently recompute scores."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def balanced(y, prediction):
    return float(np.mean([np.mean(prediction[y == label] == label) for label in np.unique(y)]))

def main():
    report_hashes = json.loads((HERE / 'results/PACKAGE_SHA256.json').read_text())
    for rel, expected in report_hashes.items():
        assert sha(HERE / 'results' / rel) == expected, rel
    checks = {}
    for name, expected_count in [('gnn_faced', 304), ('gnn_seed7', 304), ('matched_eegnet', 300)]:
        base = HERE / 'retained_runs' / name
        config = json.loads((base / 'config.json').read_text())
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        paths = sorted((base / 'runs').glob('*.json'))
        assert len(paths) == expected_count
        errors, artifact_checks = [], 0
        for path in paths:
            record = json.loads(path.read_text())
            assert record['config_sha256'] == config_hash, path
            p = base / record['predictions_path']
            if name == 'matched_eegnet':
                assert record['complete'] and record['refit_outer_train']
                assert not set(record['test_subjects']) & set(record['outer_train_subjects'])
                assert not set(record['inner_train_subjects']) & set(record['validation_subjects'])
                assert set(record['inner_train_subjects']) | set(record['validation_subjects']) == set(record['outer_train_subjects'])
                assert sha(p) == record['artifact_sha256'][record['predictions_path']]
                artifact_checks += 1
                curves = base / record['curves_path']
                assert sha(curves) == record['artifact_sha256'][record['curves_path']]
            with np.load(p, allow_pickle=False) as data:
                probability = data['probabilities'].astype(float)
                y = data['y']; pred = data['pred']
                assert np.isfinite(probability).all() and probability.min() >= -1e-7 and probability.max() <= 1+1e-7
                assert np.allclose(probability.sum(axis=1), 1, atol=2e-6, rtol=0)
                assert np.array_equal(pred, probability.argmax(axis=1))
                if name == 'matched_eegnet':
                    trials = data['trial_idx']
                    ids = np.unique(trials)
                    labels, predictions = [], []
                    for trial in ids:
                        mask = trials == trial
                        assert mask.sum() == (3 if record['seconds'] == 10 else 1)
                        assert len(np.unique(y[mask])) == 1
                        labels.append(y[mask][0]); predictions.append(probability[mask].mean(axis=0).argmax())
                    score = balanced(np.asarray(labels), np.asarray(predictions))
                    assert set(data['subject'].tolist()) == set(record['test_subjects'])
                else:
                    score = balanced(y, pred)
                errors.append(abs(score-record['balanced_accuracy']))
        assert max(errors) < 1e-12
        checks[name] = dict(runs=len(paths), original_prediction_hashes_checked=artifact_checks,
                            max_score_reconstruction_error=max(errors), config_sha256=config_hash)
    corrections = json.loads((HERE / 'provenance/CORRECTION_SOURCES_MANIFEST.json').read_text())
    for record in corrections:
        assert sha(HERE / record['copy']) == record['sha256'], record['copy']
    result = dict(status='passed', verified_original_report_files=len(report_hashes),
                  runs=checks, correction_files_checked=len(corrections),
                  scope='Stored predictions/configurations and available original artifact hashes verified; this is not a retraining check. GNN run records do not contain historical NPZ checksums; their current packaged bytes are covered by the final archive manifest.')
    (HERE / 'ARCHIVE_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
