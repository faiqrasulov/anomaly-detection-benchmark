"""Detectors implemented from scratch in numpy: HBOS, kNN distance, PCA reconstruction error.

All scores follow one convention: higher = more anomalous.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .detectors import Detector


class HBOS(Detector):
    """Histogram-based outlier score (Goldstein & Dengel, 2012).

    One static-width histogram per feature with ``n_bins`` bins (default ``sqrt(n)``) spanning
    the training range. Bin heights are normalised so the tallest bin is 1 and smoothed with
    ``alpha`` so that empty bins (only reachable by new data) get a finite height::

        h_j = (count_j + alpha) / (max_j count_j + alpha)
        score(x) = sum_over_features -log h_{bin(x)}

    Treating features independently makes HBOS linear-time and very fast, but blind to
    anomalies that are only unusual in a *combination* of features.
    """

    name = "HBOS"

    def __init__(self, n_bins: int | None = None, alpha: float = 0.1, seed: int = 0):
        super().__init__(seed)
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, X: np.ndarray) -> HBOS:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = self.n_bins or max(2, int(np.sqrt(n)))
        self.k_ = k
        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        span = self.max_ - self.min_
        self.constant_ = span == 0
        self.width_ = np.where(self.constant_, 1.0, span / k)
        self.neg_log_height_ = np.zeros((d, k + 1))  # column k = out of range / empty
        for j in range(d):
            if self.constant_[j]:
                continue
            counts = np.bincount(self._bin(X[:, j], j), minlength=k).astype(np.float64)
            height = (counts + self.alpha) / (counts.max() + self.alpha)
            self.neg_log_height_[j, :k] = -np.log(height)
            self.neg_log_height_[j, k] = -np.log(self.alpha / (counts.max() + self.alpha))
        return self

    def _bin(self, x: np.ndarray, j: int) -> np.ndarray:
        """Bin index of each value; the right edge belongs to the last bin (as np.histogram)."""
        idx = np.floor((x - self.min_[j]) / self.width_[j]).astype(np.int64)
        return np.minimum(idx, self.k_ - 1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(X.shape[0])
        for j in range(X.shape[1]):
            if self.constant_[j]:
                continue
            x = X[:, j]
            idx = self._bin(x, j)
            outside = (x < self.min_[j]) | (x > self.max_[j])
            idx[outside] = self.k_
            out += self.neg_log_height_[j, idx]
        return out


class KNNDistance(Detector):
    """Distance-based outlier score from the k nearest neighbours.

    ``method="mean"``: average distance to the k nearest neighbours (Angiulli & Pizzuti, 2002);
    ``method="largest"``: distance to the k-th neighbour (Ramaswamy et al., 2000). Training
    rows never count as their own neighbour. Only the neighbour search uses scikit-learn.
    """

    name = "kNN distance"

    def __init__(self, k: int = 10, method: str = "mean", seed: int = 0):
        super().__init__(seed)
        if method not in ("mean", "largest"):
            raise ValueError("method must be 'mean' or 'largest'")
        self.k = k
        self.method = method

    def fit(self, X: np.ndarray) -> KNNDistance:
        X = np.asarray(X, dtype=np.float64)
        self.k_ = min(self.k, X.shape[0] - 1)
        self.nn_ = NearestNeighbors(n_neighbors=self.k_ + 1).fit(X)
        return self

    def _reduce(self, dist: np.ndarray) -> np.ndarray:
        return dist.mean(axis=1) if self.method == "mean" else dist[:, -1]

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        dist, _ = self.nn_.kneighbors(np.asarray(X, dtype=np.float64), n_neighbors=self.k_)
        return self._reduce(dist)

    def train_scores(self, X: np.ndarray) -> np.ndarray:
        # Query k + 1 neighbours and drop the closest: it is the point itself or an exact
        # duplicate, both at distance 0, so the remaining distances are the same either way.
        dist, _ = self.nn_.kneighbors(np.asarray(X, dtype=np.float64), n_neighbors=self.k_ + 1)
        return self._reduce(dist[:, 1:])


class PCAReconstruction(Detector):
    """Squared reconstruction error after projecting onto the top principal components.

    Keeps the fewest components that explain ``variance`` of the total variance, but always
    leaves at least one residual direction. Anomalies that break the main correlation
    structure land far from the principal subspace (Shyu et al., 2003).
    """

    name = "PCA reconstruction"

    def __init__(self, variance: float = 0.9, seed: int = 0):
        super().__init__(seed)
        self.variance = variance

    def fit(self, X: np.ndarray) -> PCAReconstruction:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / max(X.shape[0] - 1, 1)
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]
        ratio = np.cumsum(eigval) / eigval.sum()
        q = int(np.searchsorted(ratio, self.variance) + 1)
        self.n_components_ = int(np.clip(q, 1, max(X.shape[1] - 1, 1)))
        self.components_ = eigvec[:, : self.n_components_]
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        resid = Xc - (Xc @ self.components_) @ self.components_.T
        return np.einsum("ij,ij->i", resid, resid)
