"""anomalybench: a reproducible benchmark of unsupervised anomaly detectors."""

# Load scikit-learn's OpenMP-backed extensions before anything imports PyTorch. In conda-based
# Python on macOS, PyTorch's bundled libomp loaded first plus scikit-learn's own libomp means two
# OpenMP runtimes, and the first parallel neighbour search then segfaults or deadlocks. Loading
# scikit-learn first lets PyTorch bind to the same runtime.
import sklearn.neighbors  # noqa: F401  (import order matters, see above)

__version__ = "0.1.0"
