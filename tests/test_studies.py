"""Duplicate diagnostics and de-duplication."""

import numpy as np

from anomalybench.runner import duplicate_stats, shift_stats
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


def test_shift_stats_inside_and_outside_the_principal_subspace():
    rng = np.random.default_rng(0)
    # Normal rows vary mostly along feature 0; anomalies are shifted along 0 or along 1.
    normal = rng.normal(size=(2000, 2)) * [5.0, 0.1]
    along_main = np.vstack([normal, [[30.0, 0.0]] * 20])
    along_minor = np.vstack([normal, [[0.0, 3.0]] * 20])
    y = np.r_[np.zeros(2000), np.ones(20)].astype(int)
    a = shift_stats(along_main, y, ["main", "minor"])
    b = shift_stats(along_minor, y, ["main", "minor"])
    assert a["largest_shift_feature"] == "main" and a["shift_share_in_pca_subspace"] > 0.99
    assert b["largest_shift_feature"] == "minor" and b["shift_share_in_pca_subspace"] < 0.01
    assert a["pca_components"] == 1
