from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from dataset_loaders.BAS_RVG1 import load_sp1_dataframe
from dataset_loaders.switchboard_benchmark import load_switchboard_benchmark_dataframe
from dataset_loaders.verbmobil_small import load_verbmobil_small_dataframe
from whisper_pipeline import run_whisper_large_v3_pipeline
from parakeet_pipeline import run_parakeet_pipeline
from paths import OUTPUT_PATH


def _load_bas_rvg1() -> pd.DataFrame:
    """Load BAS RVG1 SP1 data with ground-truth transcriptions."""
    return load_sp1_dataframe()

DATASET_LOADERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "bas_rvg1": _load_bas_rvg1,
    "switchboard_benchmark": load_switchboard_benchmark_dataframe,
    "verbmobil_small": load_verbmobil_small_dataframe,
}


def transcribe_dataset(
    dataset_key: str,
    model_type: str = "both",
    *,
    whisper_batch_size: Optional[int] = None,
    parakeet_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load a dataset, run ASR inference, and compute WER.

    Parameters
    ----------
    dataset_key:
        Dataset identifier.
    model_type:
        ASR model to use: "whisper", "parakeet", or "both".
    whisper_batch_size:
        Optional batch size override for Whisper inference. Defaults to 16 when not provided.
    parakeet_batch_size:
        Optional batch size override for Parakeet inference. Defaults to 16 when not provided.

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

    audio_column = "audio_source" if "audio_source" in df.columns and df["audio_source"].notna().any() else "audio_path"

    whisper_batch = whisper_batch_size if whisper_batch_size and whisper_batch_size > 0 else 16
    parakeet_batch = parakeet_batch_size if parakeet_batch_size and parakeet_batch_size > 0 else 16

    if model_type == "whisper":
        return run_whisper_large_v3_pipeline(df, audio_column=audio_column, batch_size=whisper_batch)
    elif model_type == "parakeet":
        return run_parakeet_pipeline(df, audio_column=audio_column, batch_size=parakeet_batch)
    elif model_type == "both":
        # Run both models and merge results
        df = run_whisper_large_v3_pipeline(df, audio_column=audio_column, batch_size=whisper_batch)
        df = run_parakeet_pipeline(df, audio_column=audio_column, batch_size=parakeet_batch)
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
    parser.add_argument(
        "--whisper-batch-size",
        type=int,
        default=16,
        help="Batch size for Whisper inference. Larger values improve GPU utilisation. Default: 16.",
    )
    parser.add_argument(
        "--parakeet-batch-size",
        type=int,
        default=16,
        help="Batch size for Parakeet inference. Larger values may require more GPU memory. Default: 16.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    df = transcribe_dataset(
        args.dataset,
        model_type=args.model,
        whisper_batch_size=args.whisper_batch_size,
        parakeet_batch_size=args.parakeet_batch_size,
    )

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
