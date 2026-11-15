"""Metrics: known answers, tie handling checked against exhaustive enumeration."""

from itertools import permutations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from anomalybench.metrics import (
    average_precision,
    budget_metrics,
    evaluate,
    expected_hits_at_k,
    precision_at_k,
    precision_at_n,
    recall_at_k,
    roc_auc,
)


def test_precision_at_n_known_answer():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    assert precision_at_n(y, s) == 0.5  # n = 2: top two are rows 0 (hit) and 1 (miss)
    assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(y, s, 1) == 0.5
    assert precision_at_n(y, -s) == 0.0
    assert precision_at_n(y, y.astype(float)) == 1.0


def test_ties_use_expected_value():
    # k = 2 but three rows share the top score, one of them an anomaly -> 2 * 1/3 hits.
    y = np.array([1, 0, 0, 1])
    s = np.array([1.0, 1.0, 1.0, 0.0])
    assert expected_hits_at_k(y, s, 2) == pytest.approx(2 / 3)
    assert precision_at_k(y, s, 2) == pytest.approx(1 / 3)


def test_tie_handling_matches_enumeration_of_all_orders():
    y = np.array([1, 0, 1, 0, 1, 0, 0])
    s = np.array([5, 4, 4, 4, 4, 1, 0], dtype=float)
    for k in range(1, len(y) + 1):
        # Average over every order of the tied rows (indices 1-4) of the hits in the top k.
        tied = [1, 2, 3, 4]
        totals = []
        for perm in permutations(tied):
            order = [0, *perm, 5, 6]
            totals.append(y[order[:k]].sum())
        assert expected_hits_at_k(y, s, k) == pytest.approx(np.mean(totals))


def test_budget_metrics():
    y = np.zeros(200, dtype=int)
    y[:4] = 1
    s = -np.arange(200, dtype=float)  # rows 0, 1, 2, ... ranked first
    out = budget_metrics(y, s, 0.01)  # 2 alerts
    assert out == {"alerts": 2, "precision": 1.0, "recall": 0.5}
    assert budget_metrics(y, s, 0.0001)["alerts"] == 1  # always at least one alert


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(np.array([1, 0]), np.array([0.1, 0.2]), 0)


def test_wrappers_agree_with_sklearn():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    s = rng.normal(size=300) + y
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert set(evaluate(y, s)) == {"roc_auc", "ap", "p_at_n"}


def test_roc_auc_flatters_under_imbalance():
    # 1% anomalies scored just above the bulk of the normals: ROC-AUC is high, AP is not.
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(100), np.zeros(9_900)].astype(int)
    s = np.r_[rng.normal(2.0, 1.0, 100), rng.normal(0.0, 1.0, 9_900)]
    assert roc_auc(y, s) > 0.9
    assert average_precision(y, s) < 0.3
