"""Evaluation metrics. Labels: 1 = anomaly. Scores: higher = more anomalous.

``precision_at_k`` and friends are *tie-aware*: when the k-th highest score is shared by
several rows (common with duplicated records, e.g. in KDD99), the result is the expected value
under a random choice among the tied rows, instead of whatever order ``argsort`` happens to
produce.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (ties count one half)."""
    return float(roc_auc_score(y, scores))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, i.e. the step-wise area under the precision-recall curve."""
    return float(average_precision_score(y, scores))


def expected_hits_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Expected number of anomalies among the ``k`` highest scores (random tie-breaking)."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if not 0 < k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    threshold = np.partition(scores, n - k)[n - k]  # k-th highest score
    above = scores > threshold
    tied = scores == threshold
    n_needed = k - int(above.sum())
    return float(y[above].sum() + n_needed * y[tied].sum() / tied.sum())


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of anomalies among the top ``k`` scores."""
    return expected_hits_at_k(y, scores, k) / k


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of all anomalies that appear in the top ``k`` scores."""
    return expected_hits_at_k(y, scores, k) / float(np.sum(y))


def precision_at_n(y: np.ndarray, scores: np.ndarray) -> float:
    """Precision at n = number of true anomalies (also called R-precision)."""
    return precision_at_k(y, scores, int(np.sum(y)))


def budget_metrics(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float]:
    """Alert on the top ``fraction`` of rows; return the alert count, precision and recall."""
    k = max(1, int(round(fraction * len(scores))))
    hits = expected_hits_at_k(y, scores, k)
    return {"alerts": k, "precision": hits / k, "recall": hits / float(np.sum(y))}


def evaluate(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """ROC-AUC, average precision and precision@n for one score vector."""
    return {
        "roc_auc": roc_auc(y, scores),
        "ap": average_precision(y, scores),
        "p_at_n": precision_at_n(y, scores),
    }
