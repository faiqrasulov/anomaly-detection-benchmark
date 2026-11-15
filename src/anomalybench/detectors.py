"""Detector interface and the scikit-learn based detectors.

Every detector exposes the same three calls, with one convention: **higher score = more
anomalous**.

* ``fit(X)`` learns from unlabeled data (which, in this benchmark, contains the anomalies);
* ``score_samples(X)`` scores new rows;
* ``train_scores(X)`` scores the rows the detector was fitted on. For neighbour methods this
  excludes each point from its own neighbourhood; otherwise it equals ``score_samples``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor


class Detector(ABC):
    """Common interface of all anomaly detectors in the benchmark."""

    name: ClassVar[str] = "detector"
    stochastic: ClassVar[bool] = False  # does the result depend on the seed?

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def fit(self, X: np.ndarray) -> Detector:
        """Fit on unlabeled data."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Outlier scores for new rows (higher = more anomalous)."""

    def train_scores(self, X: np.ndarray) -> np.ndarray:
        """Outlier scores for the rows passed to ``fit``."""
        return self.score_samples(X)

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and score the same rows (the unsupervised, transductive protocol)."""
        return self.fit(X).train_scores(X)


class IForest(Detector):
    """Isolation Forest (Liu, Ting & Zhou, 2008): 100 trees on subsamples of 256 rows."""

    name = "Isolation Forest"
    stochastic = True

    def __init__(self, n_estimators: int = 100, max_samples: int = 256, seed: int = 0):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X: np.ndarray) -> IForest:
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, X.shape[0]),
            random_state=self.seed,
        ).fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(X)


class LOF(Detector):
    """Local Outlier Factor (Breunig et al., 2000) with 20 neighbours."""

    name = "LOF"

    def __init__(self, n_neighbors: int = 20, seed: int = 0):
        super().__init__(seed)
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray) -> LOF:
        # novelty=True keeps score_samples() available for new data; the training-set
        # LOF values are computed in fit() either way (negative_outlier_factor_).
        self.model_ = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, X.shape[0] - 1), novelty=True
        ).fit(np.asarray(X, dtype=np.float64))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(np.asarray(X, dtype=np.float64))

    def train_scores(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.negative_outlier_factor_


class OCSVM(Detector):
    """One-Class SVM with an RBF kernel, approximated for scale.

    Nystroem features (300 components) + a linear One-Class SVM trained by SGD, so the cost is
    linear in n instead of quadratic. ``gamma`` follows scikit-learn's ``"scale"`` rule,
    ``1 / (d * Var(X))``. ``nu`` (upper bound on the fraction of training outliers) also acts as
    the regulariser, so unlike Isolation Forest's ``contamination`` it changes the scores.
    """

    name = "One-Class SVM"
    stochastic = True

    def __init__(self, nu: float = 0.05, n_components: int = 300, seed: int = 0):
        super().__init__(seed)
        self.nu = nu
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> OCSVM:
        var = float(np.var(X))
        gamma = 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        self.features_ = Nystroem(
            gamma=gamma, n_components=min(self.n_components, X.shape[0]), random_state=self.seed
        ).fit(X)
        self.model_ = SGDOneClassSVM(nu=self.nu, random_state=self.seed).fit(
            self.features_.transform(X)
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(self.features_.transform(X))


class RobustCovariance(Detector):
    """Mahalanobis distance under a robust (Minimum Covariance Determinant) Gaussian fit.

    This is scikit-learn's ``EllipticEnvelope`` score; ``contamination`` only moves its
    decision threshold, so it plays no role in the ranking metrics.
    """

    name = "Elliptic Envelope"
    stochastic = True

    def fit(self, X: np.ndarray) -> RobustCovariance:
        self.model_ = EllipticEnvelope(random_state=self.seed).fit(np.asarray(X, np.float64))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(np.asarray(X, dtype=np.float64))
