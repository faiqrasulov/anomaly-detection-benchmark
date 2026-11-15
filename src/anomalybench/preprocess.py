"""Label-free preprocessing shared by every detector.

Steps (all statistics come from the unlabeled feature matrix only):

1. drop constant numeric columns (e.g. ``num_outbound_cmds`` is always 0 in KDD99);
2. ``log1p`` for non-negative numeric columns with skewness > 2 (byte counts, durations,
   connection counts), so a handful of huge values does not set the scale of a feature;
3. robust scaling: ``(x - median) / (IQR / 1.349)``, falling back to the standard deviation
   when the IQR is 0 (columns that are mostly zero); ``1.349`` makes the IQR scale equal to the
   standard deviation for Gaussian data, so both branches live on the same scale;
4. one-hot encode categoricals as unscaled 0/1 indicators. Scaling them by their standard
   deviation would turn a rare category into a value of 30+ that dominates every distance.

The output is float32 (half the memory of float64; plenty of precision for scoring).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import skew

from .data import Dataset

SKEW_THRESHOLD = 2.0
IQR_TO_SIGMA = 1.349


@dataclass
class Preprocessed:
    """Feature matrix ready for the detectors plus a record of what was done."""

    X: np.ndarray  # (n, d) float32
    feature_names: list[str]
    log1p_columns: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    std_fallback_columns: list[str] = field(default_factory=list)
    n_onehot: int = 0


def one_hot(values: np.ndarray, name: str) -> tuple[np.ndarray, list[str]]:
    """0/1 indicator columns for each distinct value of a 1-D string array (sorted)."""
    cats, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    out = np.zeros((values.shape[0], cats.size), dtype=np.float32)
    out[np.arange(values.shape[0]), codes.ravel()] = 1.0
    return out, [f"{name}={c}" for c in cats]


def preprocess(
    numeric: np.ndarray,
    numeric_names: list[str],
    categorical: np.ndarray,
    categorical_names: list[str],
) -> Preprocessed:
    """Transform raw features into a float32 matrix. Labels are deliberately not an argument."""
    numeric = np.asarray(numeric, dtype=np.float64)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    logged, dropped, fallback = [], [], []
    for j, name in enumerate(numeric_names):
        x = numeric[:, j].copy()
        if np.ptp(x) == 0:
            dropped.append(name)
            continue
        if x.min() >= 0 and skew(x) > SKEW_THRESHOLD:
            x = np.log1p(x)
            logged.append(name)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        scale = (q3 - q1) / IQR_TO_SIGMA
        if scale <= 0:
            scale = x.std()
            fallback.append(name)
        blocks.append(((x - med) / scale)[:, None])
        names.append(name)
    n_onehot = 0
    for j, name in enumerate(categorical_names):
        block, block_names = one_hot(categorical[:, j], name)
        blocks.append(block)
        names.extend(block_names)
        n_onehot += block.shape[1]
    X = np.hstack(blocks).astype(np.float32) if blocks else np.empty((numeric.shape[0], 0))
    return Preprocessed(X.astype(np.float32), names, logged, dropped, fallback, n_onehot)


def prepare(ds: Dataset) -> tuple[Preprocessed, np.ndarray]:
    """Preprocess a dataset's features; return them with the (untouched) evaluation labels."""
    pre = preprocess(ds.numeric, ds.numeric_names, ds.categorical, ds.categorical_names)
    return pre, ds.y
