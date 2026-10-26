"""Participant isolation and coherent trial shuffling are scientific invariants."""
import unittest
import numpy as np
from common import participant_split, side_donors, trial_probabilities


class CommonTests(unittest.TestCase):
    def test_folds_cover_every_person_once_with_disjoint_validation(self):
        subject = np.repeat(np.arange(20), 12)
        covered = []
        for fold in range(5):
            split = participant_split(subject, fold)
            groups = {k: set(subject[v]) for k,v in split.items()}
            self.assertFalse(groups['inner_train'] & groups['validation'])
            self.assertFalse(groups['outer_train'] & groups['test'])
            self.assertEqual(groups['inner_train'] | groups['validation'],groups['outer_train'])
            covered.extend(groups['test'])
        self.assertEqual(sorted(covered),list(range(20)))

    def test_shuffle_keeps_participant_and_window_and_joint_trial(self):
        meta = dict(subject=np.repeat([1,2], 60),
                    trial_idx=np.repeat(np.arange(40),3),
                    window_id=np.tile(np.arange(3),40))
        rows = np.arange(120)
        donor = side_donors(meta,rows,77)
        np.testing.assert_array_equal(meta['subject'][donor],meta['subject'])
        np.testing.assert_array_equal(meta['window_id'][donor],meta['window_id'])
        for trial in range(40):
            self.assertEqual(len(np.unique(meta['trial_idx'][donor[meta['trial_idx']==trial]])),1)
        # Participant-specific permutations must not share a learnable video relabelling.
        self.assertFalse(np.array_equal(donor[:60],donor[60:]-60))
        np.testing.assert_array_equal(donor,side_donors(meta,rows,77))

    def test_trial_mean_uses_probabilities(self):
        p = np.array([[.51,.49],[.51,.49],[0,1],[.9,.1]])
        mean,y,ids = trial_probabilities(p,np.array([8,8,8,9]),np.array([1,1,1,0]))
        np.testing.assert_array_equal(mean.argmax(1),[1,0])
        with self.assertRaises(ValueError):
            trial_probabilities(p,np.array([8,8,8,9]),np.array([1,0,1,0]))


if __name__ == '__main__':
    unittest.main()
