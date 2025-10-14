#!/usr/bin/env python3
"""Recompute WER metrics for existing ASR CSV outputs.

This script updates Whisper/Parakeet WER columns against multiple reference
transcriptions:

* Ground-truth orthographic transcription (``gt_transcription``)
* Optional dialect transcription (``dialect_gt_transcription``) → ``*_wer_dialect``
* Optional Hochdeutsch translation (``hochdeutsch_translation``) → ``*_wer_hochdeutsch``

Usage
-----
    python recompute_wer.py outputs/bas_rvg1.csv
    python recompute_wer.py outputs/*.csv --glob

The script overwrites the CSV in-place. Pass ``--no-backup`` to skip creating
``.bak`` backups.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from whisper_pipeline import add_wer_column


ReferenceSpec = Tuple[str, str]


def _available_references(df: pd.DataFrame) -> list[ReferenceSpec]:
    """Return list of (column, suffix) reference pairs available in the frame."""
    candidates: tuple[ReferenceSpec, ...] = (
        ("gt_transcription", ""),  # primary ground truth
        ("dialect_gt_transcription", "dialect"),
        ("hochdeutsch_translation", "hochdeutsch"),
    )
    return [(col, suffix) for col, suffix in candidates if col in df.columns]


def _recompute_for_model(
    df: pd.DataFrame,
    *,
    transcription_column: str,
    model_prefix: str,
    references: Iterable[ReferenceSpec],
) -> pd.DataFrame:
    """Recompute WER columns for a single model/transcription column."""
    if transcription_column not in df.columns:
        print(f"  ⚠︎ Column '{transcription_column}' missing – skipping.")
        return df

    updated = df
    for reference_column, suffix in references:
        output_column = f"{model_prefix}_wer" if not suffix else f"{model_prefix}_wer_{suffix}"
        updated = add_wer_column(
            updated,
            reference_column=reference_column,
            hypothesis_column=transcription_column,
            output_column=output_column,
        )
        print(f"  ✓ recomputed {output_column} (ref='{reference_column}')")
    return updated


def recompute_csv(csv_path: Path, *, create_backup: bool = True) -> None:
    """Recompute WER columns for a single CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"Processing {csv_path} …")
    df = pd.read_csv(csv_path)
    references = _available_references(df)
    if not references:
        raise ValueError(
            f"No reference columns found in {csv_path.name}. "
            "Expected one of: gt_transcription, dialect_gt_transcription, hochdeutsch_translation."
        )

    print(f"  Reference columns: {', '.join(col for col, _ in references)}")
    df = _recompute_for_model(
        df,
        transcription_column="whisper_large_v3_transcription",
        model_prefix="whisper_large_v3",
        references=references,
    )
    df = _recompute_for_model(
        df,
        transcription_column="parakeet_tdt_v3_transcription",
        model_prefix="parakeet_tdt_v3",
        references=references,
    )

    if create_backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        backup_path.write_bytes(csv_path.read_bytes())
        print(f"  Backup written to {backup_path.name}")

    df.to_csv(csv_path, index=False)
    print("  ✓ CSV updated.\n")


def _iter_csv_paths(inputs: list[str], use_glob: bool) -> list[Path]:
    """Resolve user-specified CSV paths."""
    if not inputs:
        raise ValueError("Provide at least one CSV path or glob pattern.")

    paths: list[Path] = []
    if use_glob:
        for pattern in inputs:
            paths.extend(Path().glob(pattern))
    else:
        paths = [Path(p) for p in inputs]

    resolved = sorted(set(p.resolve() for p in paths if p.suffix.lower() == ".csv"))
    if not resolved:
        raise FileNotFoundError("No CSV files matched the provided arguments.")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute WER metrics in ASR result CSVs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="CSV file paths or glob patterns (use --glob for patterns).",
    )
    parser.add_argument(
        "--glob",
        action="store_true",
        help="Interpret provided paths as glob patterns.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before overwriting CSVs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths = _iter_csv_paths(args.paths, use_glob=args.glob)

    for csv_path in csv_paths:
        recompute_csv(csv_path, create_backup=not args.no_backup)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
