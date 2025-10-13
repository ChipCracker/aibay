#!/usr/bin/env python3
"""Enrich existing CSV outputs with Hochdeutsch translations and WER scores.

This script reads CSV files from the outputs/ directory, adds Hochdeutsch
translations using a local LLM, and calculates WER scores for both Whisper
and Parakeet models against the Hochdeutsch reference.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from hochdeutsch_pipeline import run_hochdeutsch_pipeline
from paths import OUTPUT_PATH


def backup_csv(csv_path: Path) -> Path:
    """Create a backup of the CSV file.

    Parameters
    ----------
    csv_path:
        Path to the original CSV file.

    Returns
    -------
    Path
        Path to the backup file.
    """
    backup_path = csv_path.with_suffix(".csv.backup")
    shutil.copy2(csv_path, backup_path)
    return backup_path


def enrich_csv(
    csv_path: Path,
    *,
    source_column: str = "gt_transcription",
    base_url: str = "http://localhost:1234/v1",
    model: str = "local-model",
    temperature: float = 0.3,
    max_tokens: int = 500,
    create_backup: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Enrich a single CSV file with Hochdeutsch translations and WER scores.

    Parameters
    ----------
    csv_path:
        Path to the CSV file to enrich.
    source_column:
        Column containing the dialectal ground truth text.
    base_url:
        Base URL of the local LLM API.
    model:
        Model identifier for the local LLM.
    temperature:
        Sampling temperature for translation.
    max_tokens:
        Maximum tokens for translation response.
    create_backup:
        Whether to create a backup before overwriting.
    dry_run:
        If True, don't save the enriched CSV.

    Returns
    -------
    pd.DataFrame
        The enriched DataFrame.
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Check if already enriched
    if "hochdeutsch_translation" in df.columns:
        print(f"  Warning: '{csv_path.name}' already contains Hochdeutsch translations.")
        overwrite = input("  Overwrite existing translations? (y/N): ").strip().lower()
        if overwrite != "y":
            print("  Skipping...")
            return df

    # Check if source column exists
    if source_column not in df.columns:
        print(f"  Error: Source column '{source_column}' not found in {csv_path.name}")
        print(f"  Available columns: {', '.join(df.columns)}")
        return df

    print(f"Enriching {csv_path.name} with Hochdeutsch translations and WER scores...")
    df_enriched = run_hochdeutsch_pipeline(
        df,
        source_column=source_column,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        show_progress=True,
    )

    if dry_run:
        print(f"  Dry run: Not saving changes to {csv_path.name}")
        print(f"  Preview of new columns:")
        new_columns = [
            col for col in df_enriched.columns if col not in df.columns
        ]
        print(f"    {', '.join(new_columns)}")
        return df_enriched

    # Create backup
    if create_backup:
        backup_path = backup_csv(csv_path)
        print(f"  Backup created: {backup_path.name}")

    # Save enriched CSV
    df_enriched.to_csv(csv_path, index=False)
    print(f"  Saved enriched CSV to {csv_path}")

    return df_enriched


def enrich_all_csvs(
    outputs_dir: Path,
    *,
    source_column: str = "gt_transcription",
    base_url: str = "http://localhost:1234/v1",
    model: str = "local-model",
    temperature: float = 0.3,
    max_tokens: int = 500,
    create_backup: bool = True,
    dry_run: bool = False,
    pattern: str = "*.csv",
) -> list[Path]:
    """Enrich all CSV files in the outputs directory.

    Parameters
    ----------
    outputs_dir:
        Directory containing CSV files to enrich.
    source_column:
        Column containing the dialectal ground truth text.
    base_url:
        Base URL of the local LLM API.
    model:
        Model identifier for the local LLM.
    temperature:
        Sampling temperature for translation.
    max_tokens:
        Maximum tokens for translation response.
    create_backup:
        Whether to create backups before overwriting.
    dry_run:
        If True, don't save the enriched CSVs.
    pattern:
        Glob pattern to match CSV files.

    Returns
    -------
    list[Path]
        List of enriched CSV file paths.
    """
    csv_files = list(outputs_dir.glob(pattern))

    # Filter out backup files
    csv_files = [f for f in csv_files if not f.name.endswith(".backup")]

    if not csv_files:
        print(f"No CSV files found in {outputs_dir}")
        return []

    print(f"Found {len(csv_files)} CSV file(s) to enrich:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    print()

    enriched_files = []
    for csv_file in csv_files:
        try:
            enrich_csv(
                csv_file,
                source_column=source_column,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                create_backup=create_backup,
                dry_run=dry_run,
            )
            enriched_files.append(csv_file)
            print()
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            print()

    return enriched_files


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    # Use OUTPUT_PATH from paths.py if available, otherwise default to "outputs"
    default_output_dir = Path(OUTPUT_PATH) if OUTPUT_PATH else Path("outputs")

    parser = argparse.ArgumentParser(
        description="Enrich CSV files with Hochdeutsch translations and WER scores."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        help="Specific CSV file to enrich. If not provided, all CSVs in outputs/ will be processed.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory containing CSV files (default: {default_output_dir}).",
    )
    parser.add_argument(
        "--source-column",
        type=str,
        default="gt_transcription",
        help="Column containing the dialectal ground truth text (default: gt_transcription).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="Base URL of the local LLM API (default: http://localhost:1234/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="local-model",
        help="Model identifier for the local LLM (default: local-model).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for translation (default: 0.3).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens for translation response (default: 500).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files before overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save changes, just preview what would be done.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Validate LLM API
    try:
        from hochdeutsch_pipeline import translate_to_hochdeutsch

        print("Testing connection to local LLM API...")
        test_result = translate_to_hochdeutsch(
            "Grüß Gott",
            base_url=args.base_url,
            model=args.model,
        )
        if test_result is None:
            print("Warning: LLM API test failed. Check if the server is running at", args.base_url)
            proceed = input("Continue anyway? (y/N): ").strip().lower()
            if proceed != "y":
                return 1
        else:
            print(f"  LLM API test successful: '{test_result}'")
        print()
    except Exception as e:
        print(f"Error testing LLM API: {e}")
        return 1

    # Process specific file or all files
    if args.csv_file:
        if not args.csv_file.is_file():
            print(f"Error: File not found: {args.csv_file}")
            return 1

        enrich_csv(
            args.csv_file,
            source_column=args.source_column,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            create_backup=not args.no_backup,
            dry_run=args.dry_run,
        )
    else:
        if not args.outputs_dir.is_dir():
            print(f"Error: Directory not found: {args.outputs_dir}")
            print("Run the main pipeline first to generate CSV files.")
            return 1

        enrich_all_csvs(
            args.outputs_dir,
            source_column=args.source_column,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            create_backup=not args.no_backup,
            dry_run=args.dry_run,
        )

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
