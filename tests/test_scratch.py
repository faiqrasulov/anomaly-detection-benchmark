"""From-scratch detectors: hand-computed cases, independent references, planted outliers."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from anomalybench.scratch import HBOS, KNNDistance, PCAReconstruction


def test_hbos_matches_hand_computed_example():
    # One feature, 3 bins on [0, 3]: counts [3, 2, 1]; alpha = 0.1, tallest bin -> height 1.
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [3.0]])
    det = HBOS(n_bins=3, alpha=0.1).fit(X)
    s = det.train_scores(X)
    h1, h2 = 2.1 / 3.1, 1.1 / 3.1
    np.testing.assert_allclose(s, [0, 0, 0, -np.log(h1), -np.log(h1), -np.log(h2)])
    # Out-of-range new values get the height of an empty bin: alpha / (max count + alpha).
    np.testing.assert_allclose(det.score_samples(np.array([[-1.0], [9.0]])), -np.log(0.1 / 3.1))


def test_hbos_empty_bin_and_feature_sum():
    # Feature 1: counts [2, 0, 1] -> a new value in the empty middle bin; feature 2 is constant.
    X = np.array([[0.0, 5.0], [0.0, 5.0], [3.0, 5.0]])
    det = HBOS(n_bins=3, alpha=0.1).fit(X)
    np.testing.assert_allclose(det.score_samples(np.array([[1.5, 5.0]])), -np.log(0.1 / 2.1))
    # Scores add over features: duplicating a feature doubles the score.
    X2 = np.random.default_rng(0).normal(size=(200, 1))
    one = HBOS().fit(X2).train_scores(X2)
    two = HBOS().fit(np.hstack([X2, X2])).train_scores(np.hstack([X2, X2]))
    np.testing.assert_allclose(two, 2 * one)


def test_hbos_bins_match_numpy_histogram():
    x = np.random.default_rng(1).normal(size=500)
    det = HBOS(n_bins=12).fit(x[:, None])
    counts = np.bincount(det._bin(x, 0), minlength=12)
    expected, _ = np.histogram(x, bins=12, range=(x.min(), x.max()))
    np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize(
    "detector",
    [HBOS(), KNNDistance(method="mean"), KNNDistance(method="largest"), PCAReconstruction()],
    ids=["hbos", "knn-mean", "knn-largest", "pca"],
)
def test_detects_planted_outliers(planted, detector):
    X, y = planted
    assert roc_auc_score(y, detector.fit_score(X)) > 0.9


def _brute_knn(X_ref, X_query, k, exclude_self):
    D = np.sqrt(((X_query[:, None, :] - X_ref[None, :, :]) ** 2).sum(-1))
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k]


def test_knn_matches_brute_force():
    rng = np.random.default_rng(2)
    X, Q = rng.normal(size=(60, 3)), rng.normal(size=(7, 3))
    for method, reduce in [("mean", lambda d: d.mean(1)), ("largest", lambda d: d[:, -1])]:
        det = KNNDistance(k=5, method=method).fit(X)
        np.testing.assert_allclose(det.train_scores(X), reduce(_brute_knn(X, X, 5, True)))
        np.testing.assert_allclose(det.score_samples(Q), reduce(_brute_knn(X, Q, 5, False)))


def test_knn_excludes_self_but_keeps_duplicates():
    X = np.array([[0.0], [0.0], [5.0]])
    s = KNNDistance(k=1).fit(X).train_scores(X)
    np.testing.assert_allclose(s, [0.0, 0.0, 5.0])


def test_knn_rejects_unknown_method():
    with pytest.raises(ValueError):
        KNNDistance(method="median")


def test_pca_reconstruction_matches_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 8)) + 0.2 * rng.normal(size=(300, 8))
    det = PCAReconstruction(variance=0.9).fit(X)
    ref = PCA().fit(X)
    q = int(np.searchsorted(np.cumsum(ref.explained_variance_ratio_), 0.9) + 1)
    assert det.n_components_ == q
    pca = PCA(n_components=q).fit(X)
    expected = ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)
    np.testing.assert_allclose(det.train_scores(X), expected, rtol=1e-6, atol=1e-9)


def test_pca_always_keeps_a_residual_direction():
    x = np.linspace(0, 1, 50)
    det = PCAReconstruction(variance=0.999).fit(np.column_stack([x, 2 * x]))
    assert det.n_components_ == 1
