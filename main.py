from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from datasets.BAS_RVG1 import load_sp1_dataframe
from datasets.BAS_SC1 import load_sc1_dataframe
from whisper_pipeline import run_whisper_large_v3_pipeline


def _load_bas_rvg1() -> pd.DataFrame:
    """Load BAS RVG1 SP1 data with ground-truth transcriptions."""
    return load_sp1_dataframe()

def _load_bas_sc1() -> pd.DataFrame:
    """Load BAS SC1 data with ground-truth transcriptions."""
    return load_sc1_dataframe()


DATASET_LOADERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "bas_rvg1": _load_bas_rvg1,
    "bas_sc1": _load_bas_sc1,
}


def transcribe_dataset(dataset_key: str) -> pd.DataFrame:
    """Load a dataset, run Whisper large-v3 inference, and compute WER."""
    if dataset_key not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available options: {', '.join(DATASET_LOADERS)}")

    df = DATASET_LOADERS[dataset_key]()
    if df.empty:
        raise RuntimeError(f"Dataset '{dataset_key}' returned an empty DataFrame.")

    return run_whisper_large_v3_pipeline(df)


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper large-v3 inference + WER on speech datasets.")
    parser.add_argument(
        "dataset",
        choices=sorted(DATASET_LOADERS),
        help="Dataset identifier to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV file. Defaults to outputs/<dataset>.csv.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing output to disk; results will only be printed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    df = transcribe_dataset(args.dataset)

    if args.no_write:
        print(df.head())
    else:
        output_path = args.output or Path("outputs") / f"{args.dataset}.csv"
        write_output(df, output_path)
        print(f"Wrote results to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
