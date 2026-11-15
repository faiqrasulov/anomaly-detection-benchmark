"""Run every detector on every dataset (and every seed, for stochastic detectors)."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset, subsample
from .metrics import evaluate
from .preprocess import prepare
from .registry import DETECTORS, make_detector
from .scratch import PCAReconstruction


@dataclass
class BenchmarkResult:
    """Raw per-run records, dataset statistics and (optionally) kept score vectors."""

    records: list[dict] = field(default_factory=list)
    datasets: dict[str, dict] = field(default_factory=dict)
    scores: dict[str, dict] = field(default_factory=dict)  # dataset -> {"y", det -> {seed: s}}

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.records)


def duplicate_stats(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Share of anomaly / normal rows whose feature vector occurs more than once."""
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    dup = counts[inverse.ravel()] > 1
    return {
        "dup_share_anomalies": float(dup[y == 1].mean()),
        "dup_share_normals": float(dup[y == 0].mean()),
        "distinct_anomaly_rows": int(np.unique(X[y == 1], axis=0).shape[0]),
    }


def shift_stats(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, object]:
    """Where do anomalies differ from normal rows? (A diagnostic; uses labels after the fact.)

    Reports the feature with the largest mean shift (in preprocessed, robust-scaled units) and
    the share of the squared mean-shift vector that lies inside the principal subspace a
    90%-variance PCA keeps. A share near 1 means reconstruction-based detectors (PCA, the
    autoencoder) see little of the difference.
    """
    shift = X[y == 1].mean(axis=0, dtype=np.float64) - X[y == 0].mean(axis=0, dtype=np.float64)
    j = int(np.argmax(np.abs(shift)))
    pca = PCAReconstruction(variance=0.9).fit(X)
    inside = pca.components_ @ (pca.components_.T @ shift)
    return {
        "largest_shift_feature": names[j],
        "largest_shift": float(shift[j]),
        "pca_components": pca.n_components_,
        "shift_share_in_pca_subspace": float(inside @ inside / max(shift @ shift, 1e-300)),
    }


def _short_warning(w: warnings.WarningMessage) -> str:
    return f"{w.category.__name__}: {str(w.message).splitlines()[0][:160]}"


def run_detector(name: str, X: np.ndarray, seed: int) -> tuple[np.ndarray, float, float, list]:
    """Fit + score one detector; return scores, fit seconds, score seconds and warnings."""
    det = make_detector(name, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        det.fit(X)
        t1 = time.perf_counter()
        scores = np.asarray(det.train_scores(X), dtype=np.float64)
        t2 = time.perf_counter()
    msgs = sorted({_short_warning(w) for w in caught})
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError(f"{name} produced non-finite scores")
    return scores, t1 - t0, t2 - t1, msgs


def run_benchmark(
    loaders: dict[str, Callable[[], Dataset]],
    detectors: Iterable[str] = tuple(DETECTORS),
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    max_rows: int | None = 50_000,
    keep_scores: Iterable[str] = (),
    log: Callable[[str], None] = print,
) -> BenchmarkResult:
    """Benchmark ``detectors`` on each dataset, one dataset in memory at a time.

    Stochastic detectors run once per seed; deterministic ones run once (seed = None in the
    records). Fit and score times are wall-clock seconds.
    """
    detectors, seeds, keep = list(detectors), list(seeds), set(keep_scores)
    result = BenchmarkResult()
    for ds_name, load in loaders.items():
        raw = load()
        full_n = raw.n
        ds = subsample(raw, max_rows, seed=0)
        del raw
        pre, y = prepare(ds)
        X = pre.X
        result.datasets[ds_name] = {
            "n": ds.n,
            "n_before_subsampling": full_n,
            "d_raw": len(ds.numeric_names) + len(ds.categorical_names),
            "d": int(X.shape[1]),
            "n_onehot": pre.n_onehot,
            "n_anomalies": ds.n_anomalies,
            "anomaly_rate": ds.anomaly_rate,
            "log1p_columns": pre.log1p_columns,
            "dropped_constant": pre.dropped_constant,
            "std_fallback_columns": pre.std_fallback_columns,
            "description": ds.description,
            **duplicate_stats(X, y),
            **shift_stats(X, y, pre.feature_names),
        }
        log(f"{ds_name}: n={ds.n:,} d={X.shape[1]} anomalies={ds.n_anomalies:,}")
        if ds_name in keep:
            result.scores[ds_name] = {"y": y}
        for det_name in detectors:
            run_seeds = seeds if DETECTORS[det_name].stochastic else seeds[:1]
            for seed in run_seeds:
                scores, fit_s, score_s, msgs = run_detector(det_name, X, seed)
                rec = {
                    "dataset": ds_name,
                    "detector": det_name,
                    "seed": seed if DETECTORS[det_name].stochastic else None,
                    **evaluate(y, scores),
                    "fit_s": fit_s,
                    "score_s": score_s,
                    "warnings": msgs,
                }
                result.records.append(rec)
                if ds_name in keep:
                    result.scores[ds_name].setdefault(det_name, {})[seed] = scores
            last = result.records[-1]
            log(
                f"  {det_name:12s} AP={last['ap']:.3f} ROC-AUC={last['roc_auc']:.3f} "
                f"({last['fit_s'] + last['score_s']:.2f}s per run)"
            )
    return result


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """Mean / std over seeds per (dataset, detector); times are medians over seeds."""
    g = records.groupby(["dataset", "detector"], sort=False)
    out = g.agg(
        ap=("ap", "mean"),
        ap_std=("ap", "std"),
        roc_auc=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        p_at_n=("p_at_n", "mean"),
        p_at_n_std=("p_at_n", "std"),
        fit_s=("fit_s", "median"),
        score_s=("score_s", "median"),
        runs=("ap", "size"),
    ).reset_index()
    for col in ("ap_std", "roc_auc_std", "p_at_n_std"):
        out[col] = out[col].fillna(0.0)  # deterministic detectors: one run, no spread
    return out


def metric_matrix(
    summary: pd.DataFrame, datasets: list[str], detectors: list[str], metric: str = "ap"
) -> np.ndarray:
    """(n_datasets, n_detectors) matrix of a summary metric."""
    table = summary.pivot(index="dataset", columns="detector", values=metric)
    return table.loc[datasets, detectors].to_numpy(dtype=np.float64)


def per_seed_matrix(
    records: pd.DataFrame, datasets: list[str], detectors: list[str], seeds: list[int]
) -> np.ndarray:
    """(n_seeds, n_datasets, n_detectors) AP; deterministic detectors repeat their one value."""
    out = np.empty((len(seeds), len(datasets), len(detectors)))
    for (ds, det), grp in records.groupby(["dataset", "detector"], sort=False):
        if ds not in datasets or det not in detectors:
            continue
        i, j = datasets.index(ds), detectors.index(det)
        if grp["seed"].isna().all():
            out[:, i, j] = grp["ap"].iloc[0]
        else:
            by_seed = dict(zip(grp["seed"].astype(int), grp["ap"], strict=True))
            out[:, i, j] = [by_seed[s] for s in seeds]
    return out
