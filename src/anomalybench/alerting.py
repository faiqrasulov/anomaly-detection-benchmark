"""From scores to alerts: fixed alert budgets and the ``contamination`` / ``nu`` pitfalls."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM

from .metrics import average_precision, budget_metrics, roc_auc


def budget_table(
    y: np.ndarray, scores_by_detector: dict[str, dict[int, np.ndarray]], fractions: list[float]
) -> dict[str, dict[str, dict[str, float]]]:
    """Precision / recall when alerting on the top ``fraction`` of rows, averaged over seeds.

    Returns ``{detector: {fraction: {"alerts", "precision", "recall"}}}`` (fraction as str).
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for det, per_seed in scores_by_detector.items():
        out[det] = {}
        for frac in fractions:
            rows = [budget_metrics(y, s, frac) for s in per_seed.values()]
            out[det][str(frac)] = {
                "alerts": rows[0]["alerts"],
                "precision": float(np.mean([r["precision"] for r in rows])),
                "recall": float(np.mean([r["recall"] for r in rows])),
            }
    return out


def _flag_stats(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    n_flag = int(flagged.sum())
    hits = int(y[flagged].sum())
    return {
        "flagged": n_flag,
        "flagged_share": n_flag / len(y),
        "precision": hits / n_flag if n_flag else 0.0,
        "recall": hits / int(y.sum()),
    }


def contamination_study(
    X: np.ndarray,
    y: np.ndarray,
    contaminations: list[float | str],
    nus: list[float],
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Show what the "expected anomaly share" knobs really do.

    * Isolation Forest ``contamination`` only sets the threshold of ``predict``; the scores,
      and therefore ROC-AUC and AP, are identical for every value.
    * One-Class SVM ``nu`` is also the regulariser, so it changes the fitted model and the
      ranking quality, not just the number of alerts.
    """
    iforest = []
    for c in contaminations:
        model = IsolationForest(max_samples=min(256, len(X)), contamination=c, random_state=seed)
        model.fit(X)
        scores = -model.score_samples(X)
        flagged = model.predict(X) == -1
        iforest.append(
            {
                "contamination": c,
                "ap": average_precision(y, scores),
                "roc_auc": roc_auc(y, scores),
                **_flag_stats(y, flagged),
            }
        )
    ocsvm = []
    var = float(np.var(X))
    features = Nystroem(
        gamma=1.0 / (X.shape[1] * var), n_components=min(300, len(X)), random_state=seed
    ).fit_transform(X)
    for nu in nus:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SGDOneClassSVM(nu=nu, random_state=seed).fit(features)
        scores = -model.score_samples(features)
        flagged = model.predict(features) == -1
        ocsvm.append(
            {
                "nu": nu,
                "ap": average_precision(y, scores),
                "roc_auc": roc_auc(y, scores),
                **_flag_stats(y, flagged),
            }
        )
    return {"iforest_contamination": iforest, "ocsvm_nu": ocsvm}
