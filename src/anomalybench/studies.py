"""Follow-up studies that explain the main table: duplicates in KDD99 and autoencoder training
length. Labels are used here only to *evaluate*, never to fit or to choose settings."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .autoencoder import Autoencoder
from .data import Dataset, subsample
from .metrics import average_precision
from .preprocess import prepare
from .runner import run_detector


def deduplicate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first occurrence of every distinct (features, label) record, in order."""
    _, first = np.unique(np.column_stack([X, y]), axis=0, return_index=True)
    keep = np.sort(first)
    return X[keep], y[keep]


def dedup_study(
    loaders: dict[str, Callable[[], Dataset]],
    datasets: list[str],
    detectors: list[str],
    max_rows: int | None,
    seed: int = 0,
) -> dict[str, dict]:
    """AP of every detector (one seed) after removing exact duplicate records.

    Tavallaee et al. (2009) found most KDD99 records are duplicates; here the question is how
    much they drive the ranking of unsupervised detectors.
    """
    out = {}
    for name in datasets:
        pre, y = prepare(subsample(loaders[name](), max_rows, seed=0))
        X, y = deduplicate(pre.X, y)
        aps = {det: average_precision(y, run_detector(det, X, seed)[0]) for det in detectors}
        out[name] = {"n": int(len(y)), "n_anomalies": int(y.sum()),
                     "anomaly_rate": float(y.mean()), "ap": aps}  # fmt: skip
    return out


def autoencoder_epochs_study(
    loaders: dict[str, Callable[[], Dataset]],
    datasets: list[str],
    max_rows: int | None,
    epochs: list[int],
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """AP of the autoencoder as a function of training epochs (one seed).

    In a real unsupervised deployment there are no labels to pick the epoch count, so a large
    spread here means the deep model's result is partly luck of configuration.
    """
    out = {}
    for name in datasets:
        pre, y = prepare(subsample(loaders[name](), max_rows, seed=0))
        out[name] = {
            str(e): average_precision(y, Autoencoder(epochs=e, seed=seed).fit_score(pre.X))
            for e in epochs
        }
    return out
