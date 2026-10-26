import unittest
import numpy as np
from report import validate_source_block_assignment, permutation_summary


class SourceNullReportTests(unittest.TestCase):
    manifest = {'video_to_block': [0, 0, 1, 2], 'block_labels': [0, 1, 1],
                'video_to_class': [0, 0, 1, 1], 'n_videos': 4}

    def rows(self, labels):
        return {'y': np.tile(labels, 2), 'video': np.tile(np.arange(4), 2)}

    def test_variable_clip_counts_accepted_when_blocks_preserved(self):
        observed = validate_source_block_assignment(self.rows([0, 0, 1, 1]), self.manifest, 2, 0)
        shuffled = validate_source_block_assignment(self.rows([1, 1, 0, 1]), self.manifest, 2, 1)
        self.assertFalse(np.array_equal(np.bincount(observed), np.bincount(shuffled)))

    def test_split_blocks_changed_block_counts_and_changed_observed_rejected(self):
        for labels, permutation in [([0, 1, 1, 1], 1), ([1, 1, 0, 0], 1), ([1, 1, 0, 1], 0)]:
            with self.assertRaises(ValueError):
                validate_source_block_assignment(self.rows(labels), self.manifest, 2, permutation)

    def test_participant_dependent_null_labels_rejected(self):
        data = self.rows([1, 1, 0, 1]); data['y'][4] = 0
        with self.assertRaises(ValueError):
            validate_source_block_assignment(data, self.manifest, 2, 1)

    def test_partial_null_withholds_p_value(self):
        self.assertIsNone(permutation_summary(0.9, {1: 0.4}, expected=99)['p_raw'])


if __name__ == '__main__':
    unittest.main()
