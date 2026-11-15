"""Comparing detectors across datasets: average ranks, Friedman test, Nemenyi critical
difference (Demšar, 2006) and rank stability across seeds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats


def rank_matrix(values: np.ndarray) -> np.ndarray:
    """Rank detectors within each dataset (rows = datasets, columns = detectors).

    Rank 1 = best (highest value); tied values share the average rank.
    """
    values = np.asarray(values, dtype=np.float64)
    return np.vstack([stats.rankdata(-row) for row in values])


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Mean rank of each detector over datasets."""
    return rank_matrix(values).mean(axis=0)


@dataclass
class FriedmanResult:
    """Friedman test on ranks plus the Iman-Davenport F correction."""

    chi2: float
    p_value: float
    f_stat: float
    f_p_value: float
    n_datasets: int
    n_detectors: int


def friedman_test(values: np.ndarray) -> FriedmanResult:
    """Friedman test that all detectors have the same average rank (Demšar, 2006, eq. 1-2).

    Uses the textbook statistic on average ranks, without scipy's tie correction, so it is
    the quantity the Nemenyi critical difference is defined on.
    """
    ranks = rank_matrix(values)
    n, k = ranks.shape
    r = ranks.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(r**2) - k * (k + 1) ** 2 / 4)
    p = float(stats.chi2.sf(chi2, k - 1))
    denom = n * (k - 1) - chi2
    f_stat = (n - 1) * chi2 / denom if denom > 0 else np.inf
    f_p = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1))) if np.isfinite(f_stat) else 0.0
    return FriedmanResult(float(chi2), p, float(f_stat), f_p, n, k)


def nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Critical value q_alpha of the Nemenyi test: studentized range quantile / sqrt(2)."""
    return float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference in average rank for ``k`` detectors on ``n`` datasets."""
    return nemenyi_q(k, alpha) * float(np.sqrt(k * (k + 1) / (6 * n)))


def seed_rank_stability(per_seed_values: np.ndarray) -> dict[str, object]:
    """How much the detector ordering moves when only the random seed changes.

    ``per_seed_values`` has shape (n_seeds, n_datasets, n_detectors). Returns the average rank
    of every detector under each seed, its min / max over seeds, and Kendall's tau between the
    average-rank orderings of every pair of seeds.
    """
    per_seed = np.asarray(per_seed_values, dtype=np.float64)
    avg = np.vstack([average_ranks(v) for v in per_seed])  # (n_seeds, n_detectors)
    taus = [stats.kendalltau(avg[a], avg[b]).statistic for a, b in combinations(range(len(avg)), 2)]
    winners = [
        sorted({int(np.argmax(per_seed[s, d])) for s in range(per_seed.shape[0])})
        for d in range(per_seed.shape[1])
    ]
    return {
        "avg_rank_per_seed": avg,
        "avg_rank_min": avg.min(axis=0),
        "avg_rank_max": avg.max(axis=0),
        "kendall_tau_min": float(np.min(taus)) if taus else 1.0,
        "kendall_tau_mean": float(np.mean(taus)) if taus else 1.0,
        "winners_per_dataset": winners,
    }
