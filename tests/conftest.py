"""Shared fixtures: small synthetic datasets with planted outliers (no network, no files)."""

import numpy as np
import pytest

import anomalybench  # noqa: F401  (must load before torch, see anomalybench/__init__.py)
from anomalybench.data import Dataset


def subspace_with_outliers(
    n_inliers: int = 400, n_outliers: int = 20, d: int = 6, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Inliers near a 2-D linear subspace of R^d; outliers spread widely in every direction.

    The outliers are far away (kNN), extreme in the marginals (HBOS) and off the inlier
    subspace (PCA), so every sensible detector should rank them first.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2, d))
    inliers = rng.normal(size=(n_inliers, 2)) @ W + 0.1 * rng.normal(size=(n_inliers, d))
    outliers = rng.normal(scale=4.0, size=(n_outliers, d))
    X = np.vstack([inliers, outliers]).astype(np.float32)
    y = np.r_[np.zeros(n_inliers), np.ones(n_outliers)].astype(np.int8)
    return X, y


def synthetic_dataset(
    name: str, n_inliers: int = 280, n_outliers: int = 20, seed: int = 0
) -> Dataset:
    """A raw ``Dataset`` with numeric + categorical columns (preprocessing and pipeline tests)."""
    rng = np.random.default_rng(seed)
    X, y = subspace_with_outliers(n_inliers, n_outliers, d=4, seed=seed)
    counts = rng.lognormal(mean=2.0, sigma=1.5, size=y.size)  # heavy-tailed, non-negative
    numeric = np.column_stack([X.astype(np.float64), counts, np.zeros(y.size)])  # last: constant
    proto = np.where(rng.random(y.size) < 0.7, "tcp", "udp").astype("<U4")
    # A rare category that is more common, but not exclusive, among the outliers.
    proto[(y == 1) & (rng.random(y.size) < 0.8)] = "icmp"
    proto[(y == 0) & (rng.random(y.size) < 0.03)] = "icmp"
    return Dataset(
        name,
        numeric,
        ["a", "b", "c", "d", "bytes", "always_zero"],
        proto[:, None],
        ["proto"],
        y,
        "synthetic test data",
    )


@pytest.fixture
def planted() -> tuple[np.ndarray, np.ndarray]:
    return subspace_with_outliers()
