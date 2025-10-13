from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from dataset_loaders.BAS_RVG1 import load_sp1_dataframe
from dataset_loaders.BAS_SC1 import load_sc1_dataframe
from dataset_loaders.BAS_SC10 import load_sc10_dataframe
from whisper_pipeline import run_whisper_large_v3_pipeline
from parakeet_pipeline import run_parakeet_pipeline
from paths import OUTPUT_PATH


def _load_bas_rvg1() -> pd.DataFrame:
    """Load BAS RVG1 SP1 data with ground-truth transcriptions."""
    return load_sp1_dataframe()

def _load_bas_sc1() -> pd.DataFrame:
    """Load BAS SC1 data with ground-truth transcriptions."""
    return load_sc1_dataframe()

def _load_bas_sc10() -> pd.DataFrame:
    """Load BAS SC10 data with ground-truth transcriptions."""
    return load_sc10_dataframe()


DATASET_LOADERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "bas_rvg1": _load_bas_rvg1,
    "bas_sc1": _load_bas_sc1,
    "bas_sc10": _load_bas_sc10,
}


def transcribe_dataset(dataset_key: str, model_type: str = "both") -> pd.DataFrame:
    """Load a dataset, run ASR inference, and compute WER.

    Parameters
    ----------
    dataset_key:
        Dataset identifier.
    model_type:
        ASR model to use: "whisper", "parakeet", or "both".

    Returns
    -------
    pd.DataFrame
        DataFrame with transcriptions and WER scores.
    """
    if dataset_key not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Available options: {', '.join(DATASET_LOADERS)}")

    df = DATASET_LOADERS[dataset_key]()
    if df.empty:
        raise RuntimeError(f"Dataset '{dataset_key}' returned an empty DataFrame.")

    if model_type == "whisper":
        return run_whisper_large_v3_pipeline(df)
    elif model_type == "parakeet":
        return run_parakeet_pipeline(df)
    elif model_type == "both":
        # Run both models and merge results
        df = run_whisper_large_v3_pipeline(df)
        df = run_parakeet_pipeline(df)
        return df
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: whisper, parakeet, both")


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Use OUTPUT_PATH from paths.py if available, otherwise default to "outputs"
    default_output_dir = Path(OUTPUT_PATH) if OUTPUT_PATH else Path("outputs")

    parser = argparse.ArgumentParser(description="Run ASR inference + WER on speech datasets.")
    parser.add_argument(
        "dataset",
        choices=sorted(DATASET_LOADERS),
        help="Dataset identifier to process.",
    )
    parser.add_argument(
        "--model",
        choices=["whisper", "parakeet", "both"],
        default="whisper",
        help="ASR model to use: whisper (Whisper Large V3), parakeet (Parakeet TDT v3), or both. Default: whisper.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Destination CSV file. Defaults to {default_output_dir}/<dataset>.csv.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing output to disk; results will only be printed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    df = transcribe_dataset(args.dataset, model_type=args.model)

    if args.no_write:
        print(df.head())
    else:
        # Use OUTPUT_PATH from paths.py if available, otherwise default to "outputs"
        default_output_dir = Path(OUTPUT_PATH) if OUTPUT_PATH else Path("outputs")
        output_path = args.output or default_output_dir / f"{args.dataset}.csv"
        write_output(df, output_path)
        print(f"Wrote results to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
