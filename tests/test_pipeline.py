"""End-to-end smoke test of the benchmark on tiny synthetic datasets, plus the CLI."""

import json

import numpy as np
import pytest
from conftest import synthetic_dataset

from anomalybench import cli
from anomalybench.pipeline import run_pipeline
from anomalybench.registry import DETECTORS
from anomalybench.runner import per_seed_matrix, run_benchmark, summarize


def _loaders():
    return {
        "syn-a": lambda: synthetic_dataset("syn-a", seed=0),
        "kdd-syn": lambda: synthetic_dataset("kdd-syn", n_outliers=15, seed=1),
        "syn-few": lambda: synthetic_dataset("syn-few", n_inliers=200, n_outliers=3, seed=2),
    }


def _strict_json(text: str) -> dict:
    def reject(token):
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(text, parse_constant=reject)


def test_pipeline_end_to_end(tmp_path):
    m = run_pipeline(
        _loaders(), tmp_path, seeds=[0, 1], max_rows=250, alert_dataset="syn-a",
        ae_study_min_rows=250, log=lambda _: None,
    )  # fmt: skip
    on_disk = _strict_json((tmp_path / "metrics.json").read_text())
    assert on_disk["config"]["ranked_datasets"] == ["syn-a", "kdd-syn"]
    assert on_disk["config"]["excluded_from_ranking"] == ["syn-few"]
    assert on_disk["datasets"]["syn-a"]["n"] == 250  # subsampled from 300
    n_stoch = sum(cls.stochastic for cls in DETECTORS.values())
    n_det = len(DETECTORS) - n_stoch
    assert len(on_disk["per_run"]) == 3 * (n_det + 2 * n_stoch)
    ranks = on_disk["ranking"]["avg_rank_ap"]
    assert set(ranks) == set(DETECTORS)
    assert sum(ranks.values()) == pytest.approx(len(DETECTORS) * (len(DETECTORS) + 1) / 2)
    # Isolation Forest's contamination moves only the threshold: AP never changes.
    aps = {row["ap"] for row in on_disk["contamination"]["iforest_contamination"]}
    assert len(aps) == 1
    for name in ["rocauc_vs_ap_heatmap", "mean_rank", "pr_curves_syn-a", "runtime_vs_ap"]:
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 10_000
    # Follow-up studies ran: de-duplication on the "kdd-" dataset, AE epochs on ranked ones.
    assert list(on_disk["kdd_dedup"]) == ["kdd-syn"]
    assert set(on_disk["kdd_dedup"]["kdd-syn"]["ap"]) == set(DETECTORS)
    assert set(on_disk["autoencoder_epochs"]) == {"syn-a", "kdd-syn"}
    md = (tmp_path / "results.md").read_text()
    for section in [
        "## Alert budgets on syn-a",
        "## KDD99 without duplicate records",
        "## Autoencoder: average precision vs training epochs",
    ]:
        assert section in md
    assert m["wall_time_s"] > 0


def test_runner_records_and_per_seed_matrix():
    loaders = {"syn-a": _loaders()["syn-a"]}
    res = run_benchmark(loaders, ["hbos", "iforest"], seeds=[0, 1, 2], max_rows=None,
                        keep_scores=["syn-a"], log=lambda _: None)  # fmt: skip
    df = res.frame()
    assert len(df) == 1 + 3
    assert set(res.scores["syn-a"]["iforest"]) == {0, 1, 2}
    summary = summarize(df)
    hbos = summary[summary["detector"] == "hbos"].iloc[0]
    assert hbos["runs"] == 1 and hbos["ap_std"] == 0.0
    mat = per_seed_matrix(df, ["syn-a"], ["hbos", "iforest"], [0, 1, 2])
    assert mat.shape == (3, 1, 2)
    assert np.all(mat[:, 0, 0] == mat[0, 0, 0])  # deterministic detector repeated
    np.testing.assert_allclose(mat[:, 0, 1], df[df["detector"] == "iforest"]["ap"].to_numpy())


def test_cli_rejects_unknown_dataset(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["run", "--data-home", str(tmp_path), "--datasets", "nope", "--out",
                  str(tmp_path / "r")])  # fmt: skip


def test_cli_download_failure_returns_error(tmp_path, monkeypatch, capsys):
    def offline(key, data_home):
        raise RuntimeError("Could not download (offline)")

    monkeypatch.setattr(cli, "download", offline)
    assert cli.main(["download", "--data-home", str(tmp_path)]) == 1
    assert "offline" in capsys.readouterr().err
