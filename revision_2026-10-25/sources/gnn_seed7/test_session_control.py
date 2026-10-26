import unittest
import numpy as np
from session_control import fit_predict_session


class SessionControlTests(unittest.TestCase):
    def test_balanced_rule_and_test_labels_do_not_affect_fit(self):
        # Session 1 has more class-0 trials, but is more characteristic of class 1.
        train_session = np.r_[np.ones(8), np.full(92, 2), np.ones(2), np.full(8, 2)]
        train_y = np.r_[np.zeros(100, dtype=int), np.ones(10, dtype=int)]
        session = np.r_[train_session, [1, 2]]
        y = np.r_[train_y, [0, 0]]
        train, test = np.arange(110), np.arange(110, 112)
        pred = fit_predict_session(session, y, train, test, 2)
        np.testing.assert_array_equal(pred, [1, 0])
        y[test] = 1
        np.testing.assert_array_equal(fit_predict_session(session, y, train, test, 2), pred)

    def test_unseen_session_rejected(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            fit_predict_session(np.array([1, 1, 2]), np.array([0, 1, 0]),
                                np.array([0, 1]), np.array([2]), 2)


if __name__ == '__main__':
    unittest.main()
