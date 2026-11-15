"""End-to-end reproduction: benchmark -> statistics -> reports/metrics.json, results.md, figures."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .alerting import budget_table, contamination_study
from .data import Dataset, subsample
from .figures import plot_heatmaps, plot_pr_curves, plot_ranks, plot_runtime_vs_ap
from .preprocess import prepare
from .ranking import average_ranks, friedman_test, nemenyi_cd, nemenyi_q, seed_rank_stability
from .registry import DETECTORS, FAMILY, display_name
from .report import write_results_md
from .runner import metric_matrix, per_seed_matrix, run_benchmark, summarize
from .studies import autoencoder_epochs_study, dedup_study

BUDGETS = [0.005, 0.01, 0.02, 0.05]
CONTAMINATIONS: list[float | str] = ["auto", 0.001, 0.01, 0.05, 0.1]
NUS = [0.01, 0.05, 0.1, 0.2, 0.5]
AE_EPOCHS = [5, 10, 25, 50]


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _environment(threads: int | None) -> dict[str, str]:
    import scipy
    import sklearn
    import torch

    return {
        "anomalybench": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": _cpu_name(),
        "threads": str(threads) if threads else "default",
    }


def _round(obj: object, digits: int = 4) -> object:
    """Round floats recursively so metrics.json stays readable."""
    if isinstance(obj, float | np.floating):
        return round(float(obj), digits)
    if isinstance(obj, dict):
        return {str(k): _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return _round(obj.tolist(), digits)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_pipeline(
    loaders: dict[str, Callable[[], Dataset]],
    out_dir: str | Path,
    detectors: list[str] | None = None,
    seeds: list[int] | None = None,
    max_rows: int | None = 50_000,
    alert_dataset: str = "covtype",
    min_anomalies: int = 10,
    threads: int | None = None,
    ae_study_min_rows: int = 10_000,
    log: Callable[[str], None] = print,
) -> dict:
    """Run the whole study and write ``metrics.json``, ``results.md`` and four figures."""
    t_start = time.perf_counter()
    detectors = detectors or list(DETECTORS)
    seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(
        loaders, detectors, seeds, max_rows, keep_scores=[alert_dataset], log=log
    )
    records = result.frame()
    summary = summarize(records)
    datasets = list(loaders)
    ranked = [d for d in datasets if result.datasets[d]["n_anomalies"] >= min_anomalies]
    excluded = [d for d in datasets if d not in ranked]

    ap = metric_matrix(summary, ranked, detectors, "ap")
    roc = metric_matrix(summary, ranked, detectors, "roc_auc")
    ap_all = metric_matrix(summary, datasets, detectors, "ap")
    rank_ap, rank_roc = average_ranks(ap), average_ranks(roc)
    rank_ap_all = average_ranks(ap_all)
    k, n = len(detectors), len(ranked)
    fried = friedman_test(ap) if n >= 2 else None
    cd = nemenyi_cd(k, n) if n >= 1 else float("nan")
    stability = seed_rank_stability(per_seed_matrix(records, ranked, detectors, seeds))

    # Practical part: alert budgets and contamination pitfalls on one dataset.
    alert = result.scores[alert_dataset]
    y_alert = alert["y"]
    budgets = budget_table(y_alert, {d: alert[d] for d in detectors}, BUDGETS)
    pre, y_c = prepare(subsample(loaders[alert_dataset](), max_rows, seed=0))
    contamination = contamination_study(pre.X, y_c, CONTAMINATIONS, NUS, seed=seeds[0])

    # Explaining the table: duplicates in KDD99, and the autoencoder's training length.
    kdd = [d for d in ranked if d.startswith("kdd-")]
    log("follow-up studies: KDD de-duplication, autoencoder epochs")
    dedup = dedup_study(loaders, kdd, detectors, max_rows, seed=seeds[0])
    for ds in kdd:
        dedup[ds]["ap_with_duplicates"] = {
            r["detector"]: r["ap"]
            for r in result.records
            if r["dataset"] == ds and r["seed"] in (None, seeds[0])
        }
    large = [d for d in ranked if result.datasets[d]["n"] >= ae_study_min_rows]
    ae_epochs = (
        autoencoder_epochs_study(loaders, large, max_rows, AE_EPOCHS, seed=seeds[0])
        if "autoencoder" in detectors
        else {}
    )

    runtime = {}
    for det in detectors:
        rows = summary[summary["detector"] == det]
        runtime[det] = {
            "total_s_all_datasets": float((rows["fit_s"] + rows["score_s"]).sum()),
            "mean_ap": float(rows.set_index("dataset").loc[ranked, "ap"].mean()),
        }

    metrics = {
        "config": {
            "detectors": {d: display_name(d) for d in detectors},
            "families": {d: FAMILY[d] for d in detectors},
            "seeds": seeds,
            "max_rows": max_rows,
            "alert_dataset": alert_dataset,
            "min_anomalies_for_ranking": min_anomalies,
            "ranked_datasets": ranked,
            "excluded_from_ranking": excluded,
            "budgets": BUDGETS,
        },
        "environment": _environment(threads),
        "datasets": result.datasets,
        "results": summary.to_dict(orient="records"),
        "ranking": {
            "avg_rank_ap": dict(zip(detectors, rank_ap, strict=True)),
            "avg_rank_roc_auc": dict(zip(detectors, rank_roc, strict=True)),
            "avg_rank_ap_all_datasets": dict(zip(detectors, rank_ap_all, strict=True)),
            "mean_ap": dict(zip(detectors, ap.mean(axis=0), strict=True)),
            "mean_roc_auc": dict(zip(detectors, roc.mean(axis=0), strict=True)),
            "friedman": None if fried is None else fried.__dict__,
            "nemenyi_q_0.05": nemenyi_q(k),
            "nemenyi_cd_0.05": cd,
        },
        "seed_stability": {
            "avg_rank_ap_min": dict(zip(detectors, stability["avg_rank_min"], strict=True)),
            "avg_rank_ap_max": dict(zip(detectors, stability["avg_rank_max"], strict=True)),
            "kendall_tau_min": stability["kendall_tau_min"],
            "kendall_tau_mean": stability["kendall_tau_mean"],
            "best_detector_per_dataset_across_seeds": {
                ds: [detectors[j] for j in w]
                for ds, w in zip(ranked, stability["winners_per_dataset"], strict=True)
            },
            "mean_ap_std_over_seeds": {
                d: float(summary[summary["detector"] == d]["ap_std"].mean()) for d in detectors
            },
        },
        "alert_budget": {"dataset": alert_dataset, "table": budgets},
        "contamination": {"dataset": alert_dataset, **contamination},
        "kdd_dedup": dedup,
        "autoencoder_epochs": ae_epochs,
        "runtime": runtime,
        "per_run": result.records,
    }

    # Figures.
    labels = [display_name(d) for d in detectors]
    info = result.datasets
    ds_labels = [
        f"{d}{'*' if d in excluded else ''}\n{info[d]['anomaly_rate']:.2%}"
        f"\n({info[d]['n_anomalies']:,})"
        for d in datasets
    ]
    plot_heatmaps(
        metric_matrix(summary, datasets, detectors, "roc_auc").T,
        ap_all.T,
        labels,
        ds_labels,
        fig_dir / "rocauc_vs_ap_heatmap.png",
        footnote=(
            "Column labels: anomaly rate (number of anomalies). "
            + (
                f"* fewer than {min_anomalies} anomalies: shown, but not used for ranks."
                if excluded
                else ""
            )
        ),
    )
    plot_ranks(
        rank_ap,
        stability["avg_rank_min"],
        stability["avg_rank_max"],
        labels,
        cd,
        n,
        len(seeds),
        fig_dir / "mean_rank.png",
    )
    plot_pr_curves(
        y_alert,
        {display_name(d): alert[d][seeds[0]] for d in detectors},
        budget=0.01,
        title=f"{alert_dataset} ({result.datasets[alert_dataset]['anomaly_rate']:.2%} anomalies)",
        path=fig_dir / f"pr_curves_{alert_dataset}.png",
    )
    plot_runtime_vs_ap(
        [runtime[d]["total_s_all_datasets"] for d in detectors],
        [runtime[d]["mean_ap"] for d in detectors],
        labels,
        [FAMILY[d] for d in detectors],
        fig_dir / "runtime_vs_ap.png",
    )

    metrics["wall_time_s"] = time.perf_counter() - t_start
    metrics = _round(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_results_md(metrics, out_dir / "results.md")
    log(f"wrote {out_dir / 'metrics.json'} in {metrics['wall_time_s']:.0f}s")
    return metrics
