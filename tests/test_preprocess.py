"""Preprocessing: shapes, dtypes, each transformation, and no label leakage."""

import inspect

import numpy as np
from conftest import synthetic_dataset

from anomalybench.data import Dataset
from anomalybench.preprocess import one_hot, prepare, preprocess


def test_output_shape_dtype_and_names():
    ds = synthetic_dataset("s")
    pre, y = prepare(ds)
    # 6 numeric - 1 constant + 3 one-hot (icmp, tcp, udp)
    assert pre.X.shape == (ds.n, 8)
    assert pre.X.dtype == np.float32
    assert pre.feature_names[-3:] == ["proto=icmp", "proto=tcp", "proto=udp"]
    assert pre.dropped_constant == ["always_zero"]
    assert pre.n_onehot == 3
    np.testing.assert_array_equal(y, ds.y)


def test_one_hot_rows_sum_to_one_and_are_unscaled():
    block, names = one_hot(np.array(["b", "a", "b", "c"]), "f")
    assert names == ["f=a", "f=b", "f=c"]
    np.testing.assert_array_equal(block, [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_log1p_only_for_skewed_nonnegative_columns():
    rng = np.random.default_rng(0)
    n = 2000
    numeric = np.column_stack(
        [
            rng.lognormal(0, 2, n),  # skewed, >= 0 -> log1p
            rng.normal(0, 1, n),  # symmetric -> no
            -rng.lognormal(0, 2, n),  # skewed but negative -> no
        ]
    )
    pre = preprocess(numeric, ["skewed", "normal", "negative"], np.empty((n, 0), str), [])
    assert pre.log1p_columns == ["skewed"]


def test_robust_scaling_and_std_fallback():
    rng = np.random.default_rng(1)
    n = 4000
    mostly_zero = np.zeros(n)
    mostly_zero[:40] = rng.normal(5, 1, 40)  # IQR = 0 -> standard deviation instead
    numeric = np.column_stack([rng.normal(10, 3, n), mostly_zero])
    pre = preprocess(numeric, ["x", "rare"], np.empty((n, 0), str), [])
    x = pre.X[:, 0].astype(np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    assert abs(med) < 1e-6
    assert abs((q3 - q1) / 1.349 - 1) < 1e-5
    assert pre.std_fallback_columns == ["rare"]
    np.testing.assert_allclose(pre.X[:, 1].std(), 1.0, rtol=1e-5)


def test_labels_cannot_leak_into_features():
    assert "y" not in inspect.signature(preprocess).parameters
    ds = synthetic_dataset("s")
    shuffled = Dataset(
        ds.name, ds.numeric, ds.numeric_names, ds.categorical, ds.categorical_names,
        np.random.default_rng(0).permutation(ds.y), ds.description,
    )  # fmt: skip
    a, _ = prepare(ds)
    b, _ = prepare(shuffled)
    np.testing.assert_array_equal(a.X, b.X)
    assert not any("label" in name or name == "y" for name in a.feature_names)
    # No single column is the label itself.
    for j in range(a.X.shape[1]):
        assert not np.array_equal(a.X[:, j], ds.y.astype(np.float32))


def test_preprocessing_is_deterministic():
    ds = synthetic_dataset("s", seed=3)
    np.testing.assert_array_equal(prepare(ds)[0].X, prepare(ds)[0].X)
