"""Exercise trial/window alignment and failed-cache recovery without EEG computation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from common import digest
import prepare


def fake_process_window(raw):
    marker = raw[0, 0]
    return {'signal': np.full((30, raw.shape[1] * 4 // 5), marker, dtype=np.float32),
            'connectivity': np.arange(4785, dtype=np.float32) + marker,
            'recurrence': np.arange(585, dtype=np.float32) + marker + 10000}


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = np.zeros((4, 30, 7500), dtype=np.float32)
        for trial in range(4):
            for window in range(3):
                self.raw[trial, :, window * 2500:(window + 1) * 2500] = 100 * trial + window
        np.save(self.root / 'raw.npy', self.raw)
        self.meta = dict(idx=np.arange(4), subject=np.repeat([1, 2], 2),
                         video=np.tile([0, 1], 2), y=np.tile([1, 0], 2),
                         session=np.tile([1, 2], 2), source_block=np.tile([3, 7], 2))
        np.savez_compressed(self.root / 'trials.npz', **self.meta)
        self.item = {'raw': str(self.root / 'raw.npy'), 'metadata': str(self.root / 'trials.npz'),
                     'source_sha256': {'raw': digest(self.root / 'raw.npy'),
                                       'metadata_npz': digest(self.root / 'trials.npz')}}
        self.inputs = self.root / 'inputs.json'
        self.inputs.write_text(json.dumps({'seed7': self.item}))

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, seconds=10, chunk=0):
        return SimpleNamespace(inputs=self.inputs, prepared=self.root / 'prepared',
                               dataset='seed7', seconds=seconds, chunks=2, chunk=chunk)

    def directory(self, seconds=10):
        return self.root / 'prepared' / 'seed7' / f'{seconds}s'

    def make_chunks(self, seconds=10):
        with patch.object(prepare, 'process_window', side_effect=fake_process_window), contextlib.redirect_stdout(io.StringIO()):
            for i in range(2):
                prepare.chunk(self.args(seconds, i))

    def edit_chunk(self, number, update, seconds=10):
        path = self.directory(seconds) / 'chunks' / f'chunk_{number:03d}.npz'
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        values.update(update(values))
        np.savez_compressed(path, **values)

    def test_chunk_merge_preserves_trial_order_windows_and_side_blocks(self):
        for seconds in [10, 30]:
            with self.subTest(seconds=seconds):
                self.make_chunks(seconds)
                args = self.args(seconds)
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare.merge(args)
                directory = self.directory(seconds)
                n_windows = 30 // seconds
                marker = np.array([100 * trial + window for trial in range(4)
                                   for window in range(n_windows)], dtype=np.float32)
                signal = np.load(directory / 'signal.npy')
                side = np.load(directory / 'side.npy')
                self.assertEqual(signal.shape, (4 * n_windows, 30, seconds * 200))
                np.testing.assert_array_equal(signal[:, 0, 0], marker)
                np.testing.assert_array_equal(side[:, 0], marker)
                np.testing.assert_array_equal(side[:, 4784], marker + 4784)
                np.testing.assert_array_equal(side[:, 4785], marker + 10000)
                np.testing.assert_array_equal(side[:, -1], marker + 10584)
                with np.load(directory / 'meta.npz', allow_pickle=False) as meta:
                    for key, original in self.meta.items():
                        np.testing.assert_array_equal(meta[key], np.repeat(original, n_windows))
                    np.testing.assert_array_equal(meta['trial_idx'], np.repeat(np.arange(4), n_windows))
                    np.testing.assert_array_equal(meta['window_id'], np.tile(np.arange(n_windows), 4))
                    np.testing.assert_array_equal(meta['window_start'], np.tile(seconds * np.arange(n_windows), 4))
                manifest = json.loads((directory / 'MANIFEST.json').read_text())
                for name, expected in manifest['output_sha256'].items():
                    self.assertEqual(digest(directory / name), expected)
                with patch.object(prepare, 'process_window', side_effect=AssertionError('Should reuse valid chunk')), contextlib.redirect_stdout(io.StringIO()):
                    prepare.chunk(args)
                    prepare.merge(args)

    def test_duplicate_across_chunks_cannot_declare_complete_cache(self):
        self.make_chunks()
        self.edit_chunk(1, lambda values: {'window_idx': np.arange(len(values['window_idx']))})
        with self.assertRaisesRegex(ValueError, 'coverage'):
            prepare.merge(self.args())
        self.assertFalse((self.directory() / 'MANIFEST.json').exists())

    def test_changed_chunk_provenance_and_changed_completed_arrays_rejected(self):
        self.make_chunks()
        path = self.directory() / 'chunks/chunk_001.npz'
        original = path.read_bytes()
        self.edit_chunk(1, lambda values: {'config_sha256': np.array('wrong provenance')})
        with self.assertRaisesRegex(ValueError, 'provenance'):
            prepare.merge(self.args())
        self.assertFalse((self.directory() / 'MANIFEST.json').exists())
        path.write_bytes(original)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.merge(self.args())
        array = np.load(self.directory() / 'side.npy', mmap_mode='r+')
        array[0, 0] = -999
        array.flush(); del array
        with self.assertRaisesRegex(ValueError, 'Prepared cache changed'):
            prepare.merge(self.args())
        item = dict(self.item, source_sha256={'raw': 'changed input'})
        self.inputs.write_text(json.dumps({'seed7': item}))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            prepare.chunk(self.args())


if __name__ == '__main__':
    unittest.main()
