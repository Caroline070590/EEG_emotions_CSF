"""Compare sample-space LDA directly against sklearn's dense estimator."""
import unittest

import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from linear import DualShrinkageLDA


class DualShrinkageLDATests(unittest.TestCase):
    def compare(self, X, y, test, priors=None, tolerance=2e-9):
        # Both are evaluated in float64; this only changes arithmetic precision,
        # not the features, shrinkage estimator, class priors, or model.
        X, test = np.asarray(X, dtype=np.float64), np.asarray(test, dtype=np.float64)
        expected = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=priors).fit(X, y)
        actual = DualShrinkageLDA(priors=priors).fit(X, y)
        np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual.intercept_, expected.intercept_, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual.decision_function(test), expected.decision_function(test), rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(actual.predict_proba(test), expected.predict_proba(test), rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(actual.predict(test), expected.predict(test))
        np.testing.assert_array_equal(actual.classes_, expected.classes_)
        reference_shrinkage = [ledoit_wolf_shrinkage(StandardScaler().fit_transform(X[y == c]))
                               for c in actual.classes_]
        np.testing.assert_allclose(actual.shrinkage_, reference_shrinkage, rtol=2e-11, atol=2e-11)
        return actual

    def correlated(self, counts=(23, 31, 29), p=173, seed=154):
        rng = np.random.default_rng(seed)
        y = np.repeat(np.arange(len(counts)), counts)
        loadings = rng.normal(size=(9, p))
        effects = rng.normal(scale=0.25, size=(len(counts), p))
        X = rng.normal(size=(len(y), 9)) @ loadings + rng.normal(scale=0.5, size=(len(y), p)) + effects[y]
        scales = np.exp(rng.uniform(-2, 2, size=p))
        X *= scales
        test = (rng.normal(size=(37, 9)) @ loadings + rng.normal(scale=0.5, size=(37, p))) * scales
        return X, y, test

    def test_correlated_high_dimensional_multiclass(self):
        X, y, test = self.correlated()
        model = self.compare(X, y, test)
        self.assertEqual(model.solve_space_, "sample_cholesky_woodbury")

    def test_constant_features_and_nonuniform_priors(self):
        X, y, test = self.correlated()
        X[:, :5], test[:, :5] = 4.0, 4.0
        X[:, 5], test[:, 5] = 0.0, 0.0
        self.compare(X, y, test, priors=[0.15, 0.35, 0.50])

    def test_float32_cached_inputs_match_float64_reference(self):
        X, y, test = self.correlated(p=401)
        X, test = X.astype(np.float32), test.astype(np.float32)
        actual = DualShrinkageLDA().fit(X, y)
        expected = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X.astype(np.float64), y)
        np.testing.assert_allclose(actual.predict_proba(test), expected.predict_proba(test.astype(np.float64)), atol=2e-9, rtol=2e-9)
        np.testing.assert_array_equal(actual.predict(test), expected.predict(test.astype(np.float64)))

    def test_binary_and_label_identity(self):
        X, y, test = self.correlated(counts=(31, 43))
        y = np.where(y == 0, "neutral", "emotion")
        actual = self.compare(X, y, test)
        self.assertEqual(actual.decision_function(test).shape, (len(test),))

    def test_primal_small_feature_case(self):
        X, y, test = self.correlated(p=11)
        actual = self.compare(X, y, test)
        self.assertEqual(actual.solve_space_, "primal_lstsq")

    def test_all_constant_singular_covariance(self):
        y = np.repeat(np.arange(3), [4, 6, 5])
        X = np.ones((15, 31))
        actual = self.compare(X, y, np.ones((5, 31)))
        self.assertEqual(actual.solve_space_, "singular_primal_lstsq")


if __name__ == "__main__":
    unittest.main()
