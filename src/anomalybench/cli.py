"""Command line: ``anomalybench download`` and ``anomalybench run``."""

from __future__ import annotations

import argparse
import sys
import warnings

from .data import RAW_FILES, build_cache, dataset_loaders, download
from .registry import DETECTORS


def _set_threads(n: int) -> None:
    """Pin BLAS / OpenMP and PyTorch threads: reproducible timings, no oversubscription."""
    import torch
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=n)
    torch.set_num_threads(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anomalybench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download the raw archives and cache the subsets")
    p_dl.add_argument("--data-home", default="data")

    p_run = sub.add_parser("run", help="run the full benchmark and write reports")
    p_run.add_argument("--data-home", default="data")
    p_run.add_argument("--out", default="reports")
    p_run.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="stratified subsample size per dataset (0 = no limit)",
    )
    p_run.add_argument("--seeds", type=int, default=5, help="seeds for stochastic detectors")
    p_run.add_argument("--threads", type=int, default=2)
    p_run.add_argument("--detectors", nargs="+", choices=sorted(DETECTORS), default=None)
    p_run.add_argument("--datasets", nargs="+", default=None)
    p_run.add_argument("--alert-dataset", default="covtype")

    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            for key in RAW_FILES:
                print(f"{key}: {download(key, args.data_home)}")
            for path in build_cache(args.data_home):
                print(f"cached: {path}")
            return 0

        from .pipeline import run_pipeline

        _set_threads(args.threads)
        warnings.filterwarnings("ignore", category=FutureWarning)
        loaders = dataset_loaders(args.data_home)
        if args.datasets:
            unknown = set(args.datasets) - set(loaders)
            if unknown:
                parser.error(f"unknown datasets {sorted(unknown)}; choose from {list(loaders)}")
            loaders = {k: v for k, v in loaders.items() if k in args.datasets}
        run_pipeline(
            loaders,
            args.out,
            detectors=args.detectors,
            seeds=list(range(args.seeds)),
            max_rows=args.max_rows or None,
            alert_dataset=args.alert_dataset,
            threads=args.threads,
        )
        return 0
    except RuntimeError as err:  # download / offline problems
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
