"""Ranking statistics against scipy and Demšar (2006)."""

import numpy as np
import pytest
from scipy import stats

from anomalybench.ranking import (
    average_ranks,
    friedman_test,
    nemenyi_cd,
    nemenyi_q,
    rank_matrix,
    seed_rank_stability,
)


def test_rank_matrix_best_is_one_and_ties_average():
    ranks = rank_matrix(np.array([[0.9, 0.5, 0.5, 0.1]]))
    np.testing.assert_array_equal(ranks, [[1.0, 2.5, 2.5, 4.0]])
    np.testing.assert_array_equal(average_ranks(np.array([[1, 2], [4, 3]])), [1.5, 1.5])


def test_friedman_matches_scipy_without_ties():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(9, 5)) + np.arange(5) * 0.5  # 9 datasets, 5 detectors
    res = friedman_test(values)
    ref = stats.friedmanchisquare(*values.T)
    assert res.chi2 == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    # Iman-Davenport F from its definition.
    n, k = values.shape
    f = (n - 1) * res.chi2 / (n * (k - 1) - res.chi2)
    assert res.f_stat == pytest.approx(f)
    assert res.f_p_value == pytest.approx(stats.f.sf(f, k - 1, (k - 1) * (n - 1)))


def test_nemenyi_q_matches_demsar_table():
    # Demšar (2006), Table 5(a), alpha = 0.05, k = 2..10.
    table = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
    for k, q in zip(range(2, 11), table, strict=True):
        assert nemenyi_q(k) == pytest.approx(q, abs=2e-3)


def test_nemenyi_cd_demsar_example():
    # Demšar (2006), section 3.2.2: four classifiers on 14 data sets -> CD = 1.25.
    assert nemenyi_cd(4, 14) == pytest.approx(1.25, abs=0.01)


def test_seed_stability_identical_seeds():
    one = np.random.default_rng(0).random((4, 6))
    out = seed_rank_stability(np.stack([one, one, one]))
    assert out["kendall_tau_min"] == pytest.approx(1.0)
    np.testing.assert_array_equal(out["avg_rank_min"], out["avg_rank_max"])
    assert all(len(w) == 1 for w in out["winners_per_dataset"])


def test_seed_stability_detects_swaps():
    base = np.array([[0.9, 0.5, 0.1]])
    swapped = np.array([[0.5, 0.9, 0.1]])
    out = seed_rank_stability(np.stack([base, swapped]))
    assert out["kendall_tau_min"] < 1.0
    assert out["winners_per_dataset"] == [[0, 1]]
