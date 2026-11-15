"""Report figures (matplotlib, Agg backend, 150 dpi)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .metrics import average_precision, budget_metrics  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
# Categorical slots in fixed order (validated for colour-vision deficiency); colour follows
# the detector, never its rank.
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
FAMILY_COLORS = {
    "scikit-learn": CATEGORICAL[0],
    "from scratch": CATEGORICAL[1],
    "deep (PyTorch)": CATEGORICAL[2],
}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 11.5,
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(
    roc: np.ndarray,
    ap: np.ndarray,
    detectors: list[str],
    datasets: list[str],
    path: str | Path,
    footnote: str = "",
) -> None:
    """Side-by-side ROC-AUC and AP heatmaps (rows = detectors, columns = datasets), same scale."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), sharey=True)
    panels = [
        (roc, "ROC-AUC: most detectors look strong"),
        (ap, "Average precision (PR-AUC): the view an alert queue sees"),
    ]
    for ax, (mat, title) in zip(axes, panels, strict=True):
        im = ax.imshow(mat, cmap=BLUE_RAMP, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if v > 0.55 else INK,
                )
        ax.set_xticks(range(len(datasets)), datasets, fontsize=8)
        ax.set_yticks(range(len(detectors)), detectors)
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    if footnote:
        fig.text(0.06, -0.03, footnote, fontsize=8.5, color=INK_2, ha="left")
    fig.suptitle(
        "Same scores, two metrics: ROC-AUC hides how few alerts are real (mean over seeds)",
        x=0.06,
        ha="left",
        fontsize=12.5,
    )
    _save(fig, path)


def plot_ranks(
    avg_rank: np.ndarray,
    rank_min: np.ndarray,
    rank_max: np.ndarray,
    detectors: list[str],
    cd: float,
    n_datasets: int,
    n_seeds: int,
    path: str | Path,
) -> None:
    """Average AP rank per detector, with the min-max range over seeds and the Nemenyi CD."""
    order = np.argsort(avg_rank)[::-1]  # best ends up on top
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    y = np.arange(len(order))
    ax.barh(y, avg_rank[order], height=0.55, color=CATEGORICAL[0])
    ax.hlines(y, rank_min[order], rank_max[order], color=INK, linewidth=1.6)
    for yi, idx in zip(y, order, strict=True):
        ax.text(
            rank_max[idx] + 0.08, yi, f"{avg_rank[idx]:.2f}", va="center", fontsize=9, color=INK_2
        )
    ax.set_yticks(y, [detectors[i] for i in order])
    best = float(avg_rank.min())
    top = len(order) - 0.35
    ax.plot([best, best + cd], [top, top], color=INK, linewidth=2)
    ax.plot([best, best], [top - 0.12, top + 0.12], color=INK, linewidth=2)
    ax.plot([best + cd, best + cd], [top - 0.12, top + 0.12], color=INK, linewidth=2)
    ax.text(
        best + cd / 2,
        top + 0.22,
        f"Nemenyi critical difference = {cd:.2f} ranks",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_xlim(0, len(order) + 0.6)
    ax.set_ylim(-0.6, len(order) + 0.4)
    ax.set_xlabel(
        f"Average rank by average precision over {n_datasets} datasets (1 = best)\n"
        f"black line: best to worst average rank over {n_seeds} seeds"
    )
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Which detector to start with? Average rank (lower is better)", loc="left")
    _save(fig, path)


def plot_pr_curves(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    budget: float,
    title: str,
    path: str | Path,
) -> None:
    """Precision-recall curves, with the operating point of a fixed alert budget marked."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    aps = {name: average_precision(y, s) for name, s in scores.items()}
    top = 0.0
    for color, (name, s) in zip(CATEGORICAL, scores.items(), strict=False):
        prec, rec, _ = precision_recall_curve(y, s)
        prec, rec = prec[:-1], rec[:-1]  # drop sklearn's artificial (recall 0, precision 1) end
        top = max(top, float(prec.max()))
        ax.plot(rec, prec, color=color, linewidth=1.8, label=f"{name} (AP {aps[name]:.3f})")
        point = budget_metrics(y, s, budget)
        ax.plot(
            point["recall"],
            point["precision"],
            "o",
            color=color,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    handles, labels = ax.get_legend_handles_labels()
    order = list(np.argsort([-aps[lbl.rsplit(" (AP", 1)[0]] for lbl in labels]))
    rate = float(np.mean(y))
    handles.append(ax.axhline(rate, color=INK_2, linewidth=1.2, linestyle=(0, (1, 1.5))))
    labels.append(f"random detector (precision {rate:.2%})")
    order.append(len(labels) - 1)
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=8.5,
        loc="upper right",
        title=f"dots: alert on top {budget:.0%} of scores",
        title_fontsize=8.5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.02, top + 0.06))
    ax.set_xlabel("Recall (share of anomalies caught)")
    ax.set_ylabel("Precision (share of alerts that are real)")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"Precision-recall, {title}, seed 0", loc="left")
    _save(fig, path)


def plot_runtime_vs_ap(
    runtime_s: list[float],
    mean_ap: list[float],
    detectors: list[str],
    families: list[str],
    path: str | Path,
) -> None:
    """Total fit + score time (log scale) against mean average precision."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for fam, color in FAMILY_COLORS.items():
        idx = [i for i, f in enumerate(families) if f == fam]
        if idx:
            ax.scatter(
                [runtime_s[i] for i in idx],
                [mean_ap[i] for i in idx],
                s=70,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,
                label=fam,
                zorder=3,
            )
    for x, y, name in zip(runtime_s, mean_ap, detectors, strict=True):
        ax.annotate(
            name, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=INK
        )
    ax.set_xscale("log")
    ax.set_xlabel("Fit + score time, summed over all datasets (s, log scale, one run)")
    ax.set_ylabel("Mean average precision (ranked datasets)")
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    lo, hi = min(mean_ap), max(mean_ap)
    ax.set_ylim(max(0.0, lo - 0.05), hi + 0.06)
    ax.set_xlim(min(runtime_s) / 2, max(runtime_s) * 5)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Cost vs quality per detector", loc="left")
    _save(fig, path)
