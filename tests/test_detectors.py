"""The common detector interface, the scikit-learn wrappers and the autoencoder."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from anomalybench.autoencoder import Autoencoder
from anomalybench.detectors import LOF
from anomalybench.registry import DETECTORS, display_name, make_detector


@pytest.mark.parametrize("name", list(DETECTORS))
def test_every_detector_scores_planted_outliers_higher(planted, name):
    X, y = planted
    scores = make_detector(name, seed=0).fit_score(X)
    assert scores.shape == (X.shape[0],)
    assert np.all(np.isfinite(scores))
    # The bar is deliberately modest for the weakest methods; most score > 0.95 here.
    assert roc_auc_score(y, scores) > 0.8, display_name(name)


@pytest.mark.parametrize("name", [n for n, cls in DETECTORS.items() if cls.stochastic])
def test_stochastic_detectors_are_reproducible(planted, name):
    X, _ = planted
    a = make_detector(name, seed=3).fit_score(X)
    b = make_detector(name, seed=3).fit_score(X)
    np.testing.assert_array_equal(a, b)


def test_seed_changes_isolation_forest():
    X = np.random.default_rng(0).normal(size=(300, 4))
    a = make_detector("iforest", seed=0).fit_score(X)
    b = make_detector("iforest", seed=1).fit_score(X)
    assert not np.allclose(a, b)


def test_lof_training_scores_exclude_self(planted):
    X, _ = planted
    det = LOF().fit(X)
    np.testing.assert_allclose(det.train_scores(X), -det.model_.negative_outlier_factor_)
    # Scoring the same rows as "new" data counts each point as its own neighbour: different.
    assert not np.allclose(det.train_scores(X), det.score_samples(X))


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        make_detector("nope")


def test_autoencoder_loss_decreases_on_tiny_data():
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(64, 2)) @ rng.normal(size=(2, 6))).astype(np.float32)
    det = Autoencoder(epochs=5, min_steps=200, batch_size=32, seed=0).fit(X)
    hist = det.loss_history_
    assert len(hist) == 100  # 2 batches per epoch -> 100 epochs to reach 200 steps
    assert hist[-1] < 0.25 * hist[0]
    assert det.model_.encoder[-1].out_features == 3  # bottleneck min(8, d // 2) < d


def test_autoencoder_bottleneck_is_narrower_than_input():
    for d, expected in [(1, 1), (3, 1), (10, 5), (40, 8)]:
        X = np.zeros((8, d), dtype=np.float32)
        det = Autoencoder(epochs=1, min_steps=1).fit(X)
        assert det.model_.encoder[-1].out_features == expected
