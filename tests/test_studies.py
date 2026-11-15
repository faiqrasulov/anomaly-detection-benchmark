"""Duplicate diagnostics and de-duplication."""

import numpy as np

from anomalybench.runner import duplicate_stats
from anomalybench.studies import deduplicate


def test_duplicate_stats_known_answer():
    X = np.array([[0.0], [0.0], [1.0], [2.0], [2.0], [3.0]])
    y = np.array([1, 1, 1, 0, 0, 0])
    out = duplicate_stats(X, y)
    assert out["dup_share_anomalies"] == 2 / 3
    assert out["dup_share_normals"] == 2 / 3
    assert out["distinct_anomaly_rows"] == 2


def test_deduplicate_keeps_first_copy_in_order():
    X = np.array([[3.0], [1.0], [3.0], [1.0], [1.0]])
    y = np.array([0, 1, 0, 0, 1])  # (1, label 1) appears twice, (1, label 0) once
    Xd, yd = deduplicate(X, y)
    np.testing.assert_array_equal(Xd.ravel(), [3.0, 1.0, 1.0])
    np.testing.assert_array_equal(yd, [0, 1, 0])
