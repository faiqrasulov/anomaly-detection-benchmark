"""Name -> detector class mapping used by the benchmark runner and the CLI."""

from __future__ import annotations

from .autoencoder import Autoencoder
from .detectors import LOF, OCSVM, Detector, IForest, RobustCovariance
from .scratch import HBOS, KNNDistance, PCAReconstruction

DETECTORS: dict[str, type[Detector]] = {
    "iforest": IForest,
    "lof": LOF,
    "ocsvm": OCSVM,
    "elliptic": RobustCovariance,
    "hbos": HBOS,
    "knn": KNNDistance,
    "pca": PCAReconstruction,
    "autoencoder": Autoencoder,
}

FAMILY = {
    "iforest": "scikit-learn",
    "lof": "scikit-learn",
    "ocsvm": "scikit-learn",
    "elliptic": "scikit-learn",
    "hbos": "from scratch",
    "knn": "from scratch",
    "pca": "from scratch",
    "autoencoder": "deep (PyTorch)",
}


def make_detector(name: str, seed: int = 0) -> Detector:
    """Instantiate a registered detector with its default hyperparameters."""
    try:
        cls = DETECTORS[name]
    except KeyError as err:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}") from err
    return cls(seed=seed)


def display_name(name: str) -> str:
    """Human-readable detector name."""
    return DETECTORS[name].name
