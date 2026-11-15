"""Datasets: download, memory-light parsing and the benchmark subsets.

The raw archives are the exact files scikit-learn's ``fetch_kddcup99(percent10=True)`` and
``fetch_covtype`` download (same mirror URLs and SHA-256 checksums). They are parsed here with
``pandas.read_csv`` instead of the scikit-learn fetchers, because the fetchers' first-time
parsers build Python object arrays that need well over 1.2 GB of RAM (measured: the KDD parse
was killed at a 1.2 GB cap). The subset definitions below reproduce scikit-learn's exactly;
``tests/test_data.py`` checks that against the fetchers themselves on a tiny archive.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]  # fmt: skip
KDD_CATEGORICAL = ["protocol_type", "service", "flag"]
KDD_LOGGED = ["duration", "src_bytes", "dst_bytes"]  # log(x + 0.1) in SF / http / smtp
KDD_SA_N_ABNORMAL = 3377  # number of attack rows scikit-learn draws for "SA"

COVTYPE_COLUMNS = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am",
    "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
]  # fmt: skip
COVTYPE_NORMAL, COVTYPE_ANOMALY = 2, 4  # Lodgepole Pine vs Cottonwood/Willow


@dataclass(frozen=True)
class RawFile:
    """A raw archive: local file name, download URL and SHA-256 checksum."""

    filename: str
    url: str
    sha256: str


RAW_FILES = {
    # UCI kddcup.data_10_percent.gz, scikit-learn's mirror (sklearn.datasets._kddcup99)
    "kddcup99": RawFile(
        "kddcup.data_10_percent.gz",
        "https://ndownloader.figshare.com/files/5976042",
        "8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561",
    ),
    # UCI covtype.data.gz, scikit-learn's mirror (sklearn.datasets._covtype)
    "covtype": RawFile(
        "covtype.data.gz",
        "https://ndownloader.figshare.com/files/5976039",
        "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771",
    ),
}


@dataclass
class Dataset:
    """Raw features and labels of one benchmark dataset (labels are used only for scoring)."""

    name: str
    numeric: np.ndarray  # (n, p) float64
    numeric_names: list[str]
    categorical: np.ndarray  # (n, q) str
    categorical_names: list[str]
    y: np.ndarray  # (n,) int8, 1 = anomaly
    description: str = ""

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_anomalies(self) -> int:
        return int(self.y.sum())

    @property
    def anomaly_rate(self) -> float:
        return float(self.y.mean())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(key: str, data_home: str | os.PathLike, timeout: float = 60.0) -> Path:
    """Return the path of raw archive ``key`` in ``data_home/raw``, downloading it if missing.

    Raises ``RuntimeError`` with an actionable message when offline or on a checksum mismatch.
    """
    raw = RAW_FILES[key]
    dest = Path(data_home) / "raw" / raw.filename
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(raw.url, timeout=timeout) as resp, open(part, "wb") as f:
            while block := resp.read(1 << 20):
                f.write(block)
    except (urllib.error.URLError, OSError) as err:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {raw.url} ({err}). Check the internet connection and rerun "
            f"`make data`, or place the file manually at {dest}."
        ) from err
    digest = _sha256(part)
    if digest != raw.sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {raw.url}: got {digest}, want {raw.sha256}.")
    part.rename(dest)
    return dest


def read_kdd(path: str | os.PathLike) -> pd.DataFrame:
    """Parse a KDD Cup 1999 archive (gzip CSV, 41 features + label) column-wise."""
    names = KDD_COLUMNS + ["label"]
    dtypes = {c: "category" for c in KDD_CATEGORICAL + ["label"]}
    return pd.read_csv(path, header=None, names=names, dtype=dtypes)


def _to_str(values: pd.Series | pd.Index) -> np.ndarray:
    return np.asarray(values.astype(str).to_numpy(), dtype=str)


def kdd_subset(df: pd.DataFrame, subset: str, random_state: int = 0) -> Dataset:
    """Build scikit-learn's KDD99 subset ``SA``, ``SF``, ``http`` or ``smtp`` (attacks = 1).

    Mirrors ``sklearn.datasets.fetch_kddcup99(subset=..., random_state=..., shuffle=False)``:
    SA keeps every normal connection plus 3,377 attacks drawn *with replacement*; SF keeps
    connections with ``logged_in == 1`` and applies ``log(x + 0.1)`` to duration and byte
    counts; http / smtp further restrict SF to one service and keep only those three columns.
    """
    labels = _to_str(df["label"])
    if subset == "SA":
        normal = labels == "normal."
        abnormal_idx = np.flatnonzero(~normal)
        draw = np.random.RandomState(random_state).randint(0, abnormal_idx.size, KDD_SA_N_ABNORMAL)
        rows = np.r_[np.flatnonzero(normal), abnormal_idx[draw]]
        part = df.iloc[rows]
        numeric_names = [c for c in KDD_COLUMNS if c not in KDD_CATEGORICAL]
        categorical_names = list(KDD_CATEGORICAL)
        desc = "KDD Cup 1999 (10%), all normal traffic + 3,377 attacks sampled with replacement"
    elif subset in ("SF", "http", "smtp"):
        part = df[df["logged_in"].to_numpy() == 1].copy()
        for col in KDD_LOGGED:
            part[col] = np.log(part[col].to_numpy(dtype=np.float64) + 0.1)
        numeric_names = list(KDD_LOGGED)
        categorical_names = []
        if subset == "SF":
            categorical_names = ["service"]
            desc = "KDD Cup 1999 (10%), logged-in connections"
        else:
            part = part[_to_str(part["service"]) == subset]
            desc = f"KDD Cup 1999 (10%), logged-in {subset} connections"
    else:
        raise ValueError(f"unknown KDD subset {subset!r}")
    y = (_to_str(part["label"]) != "normal.").astype(np.int8)
    numeric = part[numeric_names].to_numpy(dtype=np.float64)
    if categorical_names:
        categorical = np.stack([_to_str(part[c]) for c in categorical_names], axis=1)
    else:
        categorical = np.empty((len(part), 0), dtype=str)
    return Dataset(f"kdd-{subset}", numeric, numeric_names, categorical, categorical_names, y, desc)


def read_covtype(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Parse the 10 quantitative CoverType columns and the cover-type label (1-7)."""
    df = pd.read_csv(path, header=None, usecols=list(range(10)) + [54], dtype=np.int32)
    return df.iloc[:, :10].to_numpy(dtype=np.float64), df.iloc[:, 10].to_numpy(dtype=np.int32)


def covtype_subset(X: np.ndarray, cover: np.ndarray) -> Dataset:
    """Liu et al. (2008) ForestCover: class 2 = normal, class 4 = anomaly, 10 quantitative cols."""
    keep = (cover == COVTYPE_NORMAL) | (cover == COVTYPE_ANOMALY)
    y = (cover[keep] == COVTYPE_ANOMALY).astype(np.int8)
    return Dataset(
        "covtype",
        X[keep],
        list(COVTYPE_COLUMNS),
        np.empty((int(keep.sum()), 0), dtype=str),
        [],
        y,
        "Forest CoverType, Lodgepole Pine (normal) vs Cottonwood/Willow (anomaly)",
    )


def breast_cancer_subset(n_malignant: int = 10) -> Dataset:
    """Goldstein & Uchida (2016): all benign cases + the first ``n_malignant`` malignant ones.

    Uses the copy of the Wisconsin Diagnostic Breast Cancer data bundled with scikit-learn
    (no download).
    """
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer()
    malignant = np.flatnonzero(bunch.target == 0)[:n_malignant]  # sklearn: 0 = malignant
    rows = np.sort(np.r_[np.flatnonzero(bunch.target == 1), malignant])
    X = np.asarray(bunch.data, dtype=np.float64)[rows]
    y = (bunch.target[rows] == 0).astype(np.int8)
    return Dataset(
        "breast-cancer",
        X,
        [str(c) for c in bunch.feature_names],
        np.empty((rows.size, 0), dtype=str),
        [],
        y,
        f"WDBC, 357 benign + first {n_malignant} malignant",
    )


def subsample(ds: Dataset, max_rows: int | None, seed: int = 0) -> Dataset:
    """Stratified subsample to at most ``max_rows`` rows, preserving the anomaly rate.

    Rows keep their original order. Returns ``ds`` unchanged when it is small enough.
    """
    if max_rows is None or ds.n <= max_rows:
        return ds
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(ds.y == 1), np.flatnonzero(ds.y == 0)
    n_pos = int(round(max_rows * pos.size / ds.n))
    keep = np.sort(
        np.r_[
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=max_rows - n_pos, replace=False),
        ]
    )
    return Dataset(
        ds.name,
        ds.numeric[keep],
        ds.numeric_names,
        ds.categorical[keep],
        ds.categorical_names,
        ds.y[keep],
        ds.description + f"; stratified subsample of {max_rows:,} rows",
    )


CACHE_VERSION = 1
KDD_SUBSETS = ("SA", "SF", "http", "smtp")


def save_dataset(ds: Dataset, path: Path) -> None:
    """Store a dataset as a compressed ``.npz`` (no pickles)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        name=np.array(ds.name),
        numeric=ds.numeric,
        numeric_names=np.array(ds.numeric_names, dtype=str),
        categorical=ds.categorical,
        categorical_names=np.array(ds.categorical_names, dtype=str),
        y=ds.y,
        description=np.array(ds.description),
    )
    tmp.rename(path)


def load_dataset(path: Path) -> Dataset:
    """Inverse of :func:`save_dataset`."""
    with np.load(path, allow_pickle=False) as z:
        return Dataset(
            str(z["name"]),
            z["numeric"],
            [str(v) for v in z["numeric_names"]],
            z["categorical"],
            [str(v) for v in z["categorical_names"]],
            z["y"],
            str(z["description"]),
        )


def _cache_path(data_home: str | os.PathLike, name: str) -> Path:
    return Path(data_home) / "cache" / f"{name}.v{CACHE_VERSION}.npz"


def build_cache(data_home: str | os.PathLike) -> list[Path]:
    """Download the archives and store every benchmark subset as a small ``.npz``.

    The full KDD archive is parsed once here (the most memory-hungry step, a few hundred MB);
    benchmark runs then only load the compact subsets.
    """
    paths = []
    kdd_raw = None
    for subset in KDD_SUBSETS:
        path = _cache_path(data_home, f"kdd-{subset}")
        if not path.exists():
            if kdd_raw is None:
                kdd_raw = read_kdd(download("kddcup99", data_home))
            save_dataset(kdd_subset(kdd_raw, subset), path)
        paths.append(path)
    del kdd_raw
    path = _cache_path(data_home, "covtype")
    if not path.exists():
        save_dataset(covtype_subset(*read_covtype(download("covtype", data_home))), path)
    paths.append(path)
    return paths


def dataset_loaders(data_home: str | os.PathLike) -> dict[str, Callable[[], Dataset]]:
    """Lazy loaders for every benchmark dataset, in reporting order."""

    def cached(name: str) -> Callable[[], Dataset]:
        def load() -> Dataset:
            path = _cache_path(data_home, name)
            if not path.exists():
                build_cache(data_home)
            return load_dataset(path)

        return load

    return {
        **{f"kdd-{s}": cached(f"kdd-{s}") for s in KDD_SUBSETS},
        "covtype": cached("covtype"),
        "breast-cancer": breast_cancer_subset,
    }
