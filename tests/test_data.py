"""Data layer: subsets identical to scikit-learn's fetchers (on tiny local archives), caching,
subsampling and offline behaviour. No network access."""

import gzip
import io
import shutil
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import sklearn.datasets._covtype as sk_covtype
import sklearn.datasets._kddcup99 as sk_kdd
from sklearn.datasets import fetch_covtype, fetch_kddcup99

from anomalybench import data
from anomalybench.data import (
    KDD_COLUMNS,
    Dataset,
    breast_cancer_subset,
    covtype_subset,
    dataset_loaders,
    download,
    kdd_subset,
    load_dataset,
    read_covtype,
    read_kdd,
    save_dataset,
    subsample,
)


def _kdd_archive(path: Path, n: int = 80, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    labels = ["normal.", "normal.", "normal.", "smurf.", "back.", "neptune."]
    services = ["http", "smtp", "ftp_data", "private"]
    lines = []
    for _ in range(n):
        row = []
        for name in KDD_COLUMNS:
            if name == "protocol_type":
                row.append(str(rng.choice(["tcp", "udp", "icmp"])))
            elif name == "service":
                row.append(str(rng.choice(services)))
            elif name == "flag":
                row.append(str(rng.choice(["SF", "S0", "REJ"])))
            elif name == "logged_in":
                row.append(str(int(rng.random() < 0.6)))
            elif "rate" in name:
                row.append(f"{rng.random():.2f}")
            else:
                row.append(str(int(rng.integers(0, 5000))))
        row.append(str(rng.choice(labels)))
        lines.append(",".join(row))
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _covtype_archive(path: Path, n: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        quant = rng.integers(0, 3500, size=10)
        binary = np.zeros(44, dtype=int)
        binary[rng.integers(0, 4)] = 1
        binary[4 + rng.integers(0, 40)] = 1
        cover = rng.choice([1, 2, 2, 3, 4, 4, 7])
        lines.append(",".join(map(str, [*quant, *binary, cover])))
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _fake_fetch_remote(archive_path: Path):
    def fake(remote, dirname=None, **kwargs):
        dest = Path(dirname) / remote.filename
        shutil.copy(archive_path, dest)
        return str(dest)

    return fake


@pytest.mark.parametrize("subset", ["SA", "SF", "http", "smtp"])
def test_kdd_subsets_identical_to_sklearn_fetcher(tmp_path, monkeypatch, subset):
    archive = _kdd_archive(tmp_path / "tiny_kdd.gz")
    monkeypatch.setattr(sk_kdd, "_fetch_remote", _fake_fetch_remote(archive))
    ref = fetch_kddcup99(subset=subset, data_home=str(tmp_path / "sk"), random_state=0)
    ours = kdd_subset(read_kdd(archive), subset, random_state=0)

    names = list(ref.feature_names)
    assert sorted(ours.numeric_names + ours.categorical_names) == sorted(names)
    for j, name in enumerate(ours.numeric_names):
        expected = ref.data[:, names.index(name)].astype(np.float64)
        np.testing.assert_allclose(ours.numeric[:, j], expected, rtol=0, atol=1e-12)
    for j, name in enumerate(ours.categorical_names):
        expected = [v.decode() for v in ref.data[:, names.index(name)]]
        assert list(ours.categorical[:, j]) == expected
    np.testing.assert_array_equal(ours.y, (ref.target != b"normal.").astype(np.int8))
    if subset == "SA":
        assert ours.n_anomalies == data.KDD_SA_N_ABNORMAL  # attacks drawn with replacement


def test_covtype_subset_identical_to_sklearn_fetcher(tmp_path, monkeypatch):
    archive = _covtype_archive(tmp_path / "tiny_covtype.gz")
    monkeypatch.setattr(sk_covtype, "_fetch_remote", _fake_fetch_remote(archive))
    ref = fetch_covtype(data_home=str(tmp_path / "sk"))
    keep = (ref.target == 2) | (ref.target == 4)
    ours = covtype_subset(*read_covtype(archive))
    np.testing.assert_array_equal(ours.numeric, ref.data[keep, :10])
    np.testing.assert_array_equal(ours.y, (ref.target[keep] == 4).astype(np.int8))
    assert ours.numeric_names[0] == "Elevation"


def test_breast_cancer_subset_is_bundled():
    ds = breast_cancer_subset()
    assert (ds.n, ds.n_anomalies, ds.numeric.shape[1]) == (367, 10, 30)


def test_subsample_is_stratified_and_ordered():
    y = np.r_[np.zeros(950), np.ones(50)].astype(np.int8)
    ds = Dataset("t", np.arange(1000.0)[:, None], ["i"], np.empty((1000, 0), str), [], y)
    sub = subsample(ds, 200, seed=0)
    assert sub.n == 200 and sub.n_anomalies == 10
    assert np.all(np.diff(sub.numeric[:, 0]) > 0)  # original order kept
    assert subsample(ds, 5000) is ds


def test_cache_roundtrip_and_loader_without_network(tmp_path, monkeypatch):
    ds = Dataset(
        "kdd-http", np.random.default_rng(0).normal(size=(5, 3)), ["a", "b", "c"],
        np.array([["x"], ["y"], ["x"], ["x"], ["z"]]), ["cat"], np.array([0, 1, 0, 0, 1], np.int8),
        "desc",
    )  # fmt: skip
    path = data._cache_path(tmp_path, "kdd-http")
    save_dataset(ds, path)
    back = load_dataset(path)
    np.testing.assert_array_equal(back.numeric, ds.numeric)
    np.testing.assert_array_equal(back.categorical, ds.categorical)
    assert (back.name, back.numeric_names, back.description) == (ds.name, ds.numeric_names, "desc")

    def no_network(*args, **kwargs):
        raise AssertionError("tried to download")

    monkeypatch.setattr(data, "download", no_network)
    loaded = dataset_loaders(tmp_path)["kdd-http"]()
    np.testing.assert_array_equal(loaded.y, ds.y)


def test_download_offline_gives_clear_error(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(data.urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="internet connection"):
        download("covtype", tmp_path)
    assert not any((tmp_path / "raw").iterdir())  # no partial file left behind


def test_download_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"junk"))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download("kddcup99", tmp_path)
