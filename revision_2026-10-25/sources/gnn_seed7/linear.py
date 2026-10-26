"""Exact automatic-shrinkage LDA using a sample-space covariance solve.

This reproduces the estimator of sklearn's
LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), with float64
arithmetic, classwise StandardScaler and Ledoit-Wolf shrinkage. No features are
discarded. The pooled covariance is represented as diagonal + U.T @ U; its
inverse action is evaluated by the Woodbury identity when samples < features.
"""
import warnings

import numpy as np
from scipy import linalg
from scipy.special import expit, softmax
from sklearn.preprocessing import StandardScaler


def _ledoit_wolf_from_samples(standardized):
    """Return (shrinkage, target mean variance, centered standardized data).

    Frobenius norm equality ||Z.T Z||_F = ||Z Z.T||_F and the squared row
    norms give sklearn's Ledoit-Wolf beta/delta without feature-square arrays.
    """
    Z = np.asarray(standardized, dtype=np.float64)
    Z = Z - Z.mean(axis=0)
    n, p = Z.shape
    squared = Z * Z
    covariance_trace = squared.sum(axis=0) / n
    mu = covariance_trace.sum() / p
    if p == 1:
        return 0.0, float(mu), Z
    gram = Z @ Z.T
    delta_raw = np.sum(gram * gram) / (n * n)
    row_norms = squared.sum(axis=1)
    beta_raw = np.dot(row_norms, row_norms)
    beta = (beta_raw / n - delta_raw) / (p * n)
    delta = (delta_raw - 2.0 * mu * covariance_trace.sum() + p * mu * mu) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0.0 else beta / delta
    if not np.isfinite(shrinkage) or shrinkage < -1e-10 or shrinkage > 1.0 + 1e-10:
        raise FloatingPointError("Invalid Ledoit-Wolf shrinkage from sample-space moments")
    return float(shrinkage), float(mu), Z


class DualShrinkageLDA:
    """Float64-equivalent sklearn lsqr/auto LDA, optionally with fixed priors.

    ``fit(X, y)`` exposes sklearn-compatible means_, priors_, classes_, coef_
    and intercept_. ``shrinkage_`` contains one estimated coefficient per
    class. ``solve_space_`` records whether the exact sample-space, primal,
    or singular least-squares path was used. The dense covariance is not kept.
    """
    def __init__(self, priors=None):
        self.priors = priors

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(X) < 2:
            raise ValueError("Expected a two-dimensional feature matrix and aligned labels")
        if X.shape[1] < 1 or not np.isfinite(X).all():
            raise ValueError("Features must be nonempty and finite")
        self.classes_, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
        n, p = X.shape
        k = len(self.classes_)
        if k < 2 or n <= k:
            raise ValueError("At least two classes and more samples than classes are required")
        if self.priors is None:
            self.priors_ = counts.astype(np.float64) / n
        else:
            self.priors_ = np.asarray(self.priors, dtype=np.float64).copy()
            if self.priors_.shape != (k,) or not np.isfinite(self.priors_).all():
                raise ValueError("One finite prior is required for each class")
            if np.any(self.priors_ < 0) or self.priors_.sum() <= 0:
                raise ValueError("Priors must be nonnegative with a positive sum")
            if abs(self.priors_.sum() - 1.0) > 1e-5:
                warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
                self.priors_ /= self.priors_.sum()
        self.n_features_in_ = p
        self.means_ = np.zeros((k, p), dtype=np.float64)
        np.add.at(self.means_, encoded, X)
        self.means_ /= counts[:, None]
        self.shrinkage_ = np.empty(k, dtype=np.float64)
        diagonal = np.zeros(p, dtype=np.float64)
        U = np.empty((n, p), dtype=np.float64)
        start = 0
        for c, count in enumerate(counts):
            values = X[encoded == c]
            scaler = StandardScaler()
            standardized = scaler.fit_transform(values)
            shrinkage, mu, centered = _ledoit_wolf_from_samples(standardized)
            self.shrinkage_[c] = shrinkage
            prior = self.priors_[c]
            diagonal += prior * shrinkage * mu * scaler.scale_ ** 2
            # This is the classwise standardized covariance rescaled exactly
            # as in sklearn.discriminant_analysis._cov(shrinkage='auto').
            U[start:start + count] = centered * scaler.scale_ * np.sqrt(prior * (1.0 - shrinkage) / count)
            start += count

        if p <= n or np.any(diagonal <= 0):
            covariance = U.T @ U
            covariance.flat[::p + 1] += diagonal
            # lstsq also reproduces sklearn for singular/constant-only inputs.
            coefficients = linalg.lstsq(covariance, self.means_.T, check_finite=False)[0].T
            self.solve_space_ = "primal_lstsq" if p <= n else "singular_primal_lstsq"
        else:
            inverse_sqrt = 1.0 / np.sqrt(diagonal)
            # Replace U in place to keep peak memory bounded.
            U *= inverse_sqrt
            system = U @ U.T
            system.flat[::n + 1] += 1.0
            factor = linalg.cho_factor(system, lower=True, overwrite_a=True, check_finite=False)
            scaled_means = self.means_ * inverse_sqrt
            rhs = U @ scaled_means.T
            correction = linalg.cho_solve(factor, rhs, check_finite=False)
            coefficients = (scaled_means - (U.T @ correction).T) * inverse_sqrt
            self.solve_space_ = "sample_cholesky_woodbury"
        self.coef_ = coefficients
        with np.errstate(divide="ignore"):
            self.intercept_ = -0.5 * np.sum(self.means_ * self.coef_, axis=1) + np.log(self.priors_)
        self.covariance_diagonal_target_ = diagonal
        if k == 2:
            self.coef_ = (self.coef_[1] - self.coef_[0])[None, :]
            self.intercept_ = np.array([self.intercept_[1] - self.intercept_[0]])
        return self

    def decision_function(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("Fit the classifier before prediction")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite with the fitted dimension")
        score = X @ self.coef_.T + self.intercept_
        return score[:, 0] if len(self.classes_) == 2 else score

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            positive = expit(scores)
            return np.column_stack([1.0 - positive, positive])
        return softmax(scores, axis=1)

    def predict(self, X):
        scores = self.decision_function(X)
        indices = (scores > 0).astype(np.int64) if len(self.classes_) == 2 else scores.argmax(axis=1)
        return self.classes_[indices]
