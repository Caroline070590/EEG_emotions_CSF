"""Protect cache/label alignment before expensive matched EEGNet extraction."""
import unittest

import numpy as np
import pandas as pd

from build_inputs import _remote_path, validate_metadata
from common import DATASETS


def frame_for(dataset):
    spec = DATASETS[dataset]
    ns, nv = spec['n_subjects'], spec['n_videos']
    counts = [4 if x == 'neutral' else 3 for x in spec['class_names']] if dataset == 'faced' else [
        8 if x == 'neutral' else 12 for x in spec['class_names']]
    emotion = np.repeat(spec['class_names'], counts)
    if dataset == 'seed7':
        emotion = np.random.default_rng(71).permutation(emotion)
    offset = 0 if dataset == 'faced' else 1
    frame = pd.DataFrame({'idx': np.arange(ns * nv),
                          'subject': np.repeat(np.arange(offset, ns + offset), nv),
                          'video': np.tile(np.arange(nv), ns),
                          'emotion': np.tile(emotion, ns)})
    if dataset == 'seed7':
        frame['clip'] = frame['video'] + 1
        frame['session'] = np.tile(np.repeat(np.arange(1, 5), 20), ns)
        frame['source'] = ['programme ' + str(v // 2) for v in frame['video']]
        frame['title'] = ['clip ' + str(v) for v in frame['video']]
    return frame


class InputTests(unittest.TestCase):
    def test_exact_row_and_class_alignment_both_datasets(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                frame = frame_for(dataset)
                raw_shape = (len(frame), 30, 7500)
                arrays, summary = validate_metadata(dataset, frame, raw_shape, 'float32')
                self.assertTrue(all(x.shape == (len(frame),) for x in arrays.values()))
                for name in ['idx', 'subject', 'video']:
                    np.testing.assert_array_equal(arrays[name], frame[name])
                np.testing.assert_array_equal(np.asarray(summary['class_names'])[arrays['y']], frame['emotion'])
                np.testing.assert_array_equal(summary['class_counts'], np.bincount(arrays['y']))
                if dataset == 'seed7':
                    np.testing.assert_array_equal(arrays['source_block'], np.tile(summary['video_to_block'], 20))
                    np.testing.assert_array_equal(np.asarray(summary['block_labels'])[arrays['source_block']], arrays['y'])

    def test_no_silent_reordering_or_duplicate_trials(self):
        original = frame_for('faced')
        swapped_rows = original.iloc[[1, 0] + list(range(2, len(original)))].copy()
        reordered_identifiers = original.copy()
        reordered_identifiers.loc[[0, 1], 'video'] = [1, 0]
        duplicate_trial = original.copy()
        duplicate_trial.loc[1, 'video'] = 0
        for bad in [swapped_rows, reordered_identifiers, duplicate_trial, original.iloc[:-1]]:
            with self.assertRaises(ValueError):
                validate_metadata('faced', bad, (len(original), 30, 7500), 'float32')

    def test_unknown_inconsistent_and_relabelled_emotions_rejected(self):
        original = frame_for('faced')
        unknown = original.copy(); unknown.loc[0, 'emotion'] = 'other'
        inconsistent = original.copy(); inconsistent.loc[0, 'emotion'] = 'fear'
        changed_mapping = original.copy()
        changed_mapping['emotion'] = changed_mapping['emotion'].replace({'anger': 'fear', 'fear': 'anger'})
        for bad in [unknown, inconsistent, changed_mapping]:
            with self.assertRaises(ValueError):
                validate_metadata('faced', bad, (len(original), 30, 7500), 'float32')

    def test_seed_independent_clip_table_rejects_consistent_wrong_relabelling(self):
        original = frame_for('seed7')
        clips = original.iloc[:80][['clip', 'session', 'emotion', 'source', 'title']].copy()
        arrays, _ = validate_metadata('seed7', original, (1600, 30, 7500), 'float32', clips)
        self.assertEqual(arrays['subject'][0], 1)
        bad = original.copy()
        bad['emotion'] = bad['emotion'].replace({'anger': 'fear', 'fear': 'anger'})
        with self.assertRaisesRegex(ValueError, 'clip table'):
            validate_metadata('seed7', bad, (1600, 30, 7500), 'float32', clips)
        bad = original.copy(); bad.loc[0, 'source'] = 'different programme'
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            validate_metadata('seed7', bad, (1600, 30, 7500), 'float32', clips)

    def test_invalid_raw_layout_and_identifiers(self):
        frame = frame_for('faced')
        for shape, dtype in [((3444, 31, 7500), 'float32'), ((3444, 30, 2500), 'float32'),
                             ((3444, 30, 7500), 'float64')]:
            with self.assertRaises(ValueError):
                validate_metadata('faced', frame, shape, dtype)
        bad = frame.copy(); bad['idx'] = bad['idx'].astype(float); bad.loc[0, 'idx'] = .5
        with self.assertRaisesRegex(ValueError, 'Noninteger'):
            validate_metadata('faced', bad, (3444, 30, 7500), 'float32')
        for path in ['relative/path', '/home/users/../other']:
            with self.assertRaises(ValueError):
                _remote_path(path)


if __name__ == '__main__':
    unittest.main()
