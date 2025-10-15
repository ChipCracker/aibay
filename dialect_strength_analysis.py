#!/usr/bin/env python3
"""Estimate dialect strength from BAS RVG1 transcriptions and correlate with WER.

The script expects a CSV that contains at least the columns:

    speaker_id, gt_transcription, dialect_gt_transcription, <wer columns>

Typical input is the combined inference output (e.g. ``outputs/bas_rvg1.csv``).
For every segment it computes a dialect-strength score defined as the share of
tokens in the ground-truth transcription that are *not* present in the dialect
transcription. Missing dialect transcriptions yield a strength of ``0``.

The result is aggregated per speaker (mean/median/max/share_nonzero) and merged
with per-speaker WER averages. Pearson and Spearman correlations between
dialect strength and WER are reported for each available model.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

TOKEN_PATTERN = re.compile(r"[\\wäöüß]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Return lower-cased tokens (unicode word characters incl. umlauts)."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def compute_dialect_strength(gt: str | float, dialect: str | float) -> float:
    """Compute share of GT tokens replaced by dialect tokens."""
    if not isinstance(gt, str) or not gt.strip():
        return math.nan
    if not isinstance(dialect, str) or not dialect.strip():
        return 0.0

    gt_tokens = tokenize(gt)
    if not gt_tokens:
        return math.nan

    dialect_tokens = tokenize(dialect)
    if not dialect_tokens:
        return 0.0

    gt_counter = Counter(gt_tokens)
    dialect_counter = Counter(dialect_tokens)
    common = sum((gt_counter & dialect_counter).values())
    ratio = 1.0 - (common / len(gt_tokens))
    return max(0.0, min(1.0, ratio))


def add_dialect_strength(df: pd.DataFrame) -> pd.DataFrame:
    strengths = [
        compute_dialect_strength(gt, dial)
        for gt, dial in zip(df["gt_transcription"], df["dialect_gt_transcription"])
    ]
    df = df.copy()
    df["dialect_strength"] = strengths
    return df


def aggregate_by_speaker(df: pd.DataFrame, wer_columns: Iterable[str]) -> pd.DataFrame:
    agg_map = {
        "dialect_strength": ["mean", "median", "max"],
    }
    for col in wer_columns:
        agg_map[col] = ["mean", "median"]

    grouped = df.groupby("speaker_id").agg(agg_map)
    grouped.columns = ["_".join(filter(None, col)).strip("_") for col in grouped.columns]

    # Share of segments with dialect influence (>0)
    share_non_zero = (
        df.assign(dialect_has_variant=lambda x: x["dialect_strength"] > 0)
        .groupby("speaker_id")["dialect_has_variant"]
        .mean()
        .rename("dialect_strength_share_nonzero")
    )
    grouped = grouped.join(share_non_zero, how="left")

    return grouped.reset_index()


def correlation_report(speaker_df: pd.DataFrame, wer_columns: Iterable[str]) -> list[str]:
    lines: list[str] = []
    strength_col = "dialect_strength_mean"

    if strength_col not in speaker_df.columns:
        return ["WARN: dialect_strength_mean missing in speaker-level dataframe."]

    for col in wer_columns:
        if col not in speaker_df.columns:
            continue

        pearson = speaker_df[strength_col].corr(speaker_df[col], method="pearson")
        spearman = speaker_df[strength_col].corr(speaker_df[col], method="spearman")

        lines.append(
            f"{col}: pearson={pearson:.4f} spearman={spearman:.4f} "
            f"(n={speaker_df[[strength_col, col]].dropna().shape[0]})"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dialect-strength metrics and correlate with WER."
    )
    parser.add_argument("input", type=Path, help="CSV with GT, dialect, and WER columns.")
    parser.add_argument("--segment-output", type=Path, help="Optional path to write segment-level CSV.")
    parser.add_argument("--speaker-output", type=Path, help="Optional path to write speaker-level CSV.")
    parser.add_argument("--no-write", action="store_true", help="Skip writing CSV outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)

    required_cols = {"speaker_id", "gt_transcription", "dialect_gt_transcription"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Keep only rows with GT available
    df = df[df["gt_transcription"].notna()].copy()

    df = add_dialect_strength(df)

    wer_columns = [col for col in df.columns if col.endswith("_wer")]
    if not wer_columns:
        print("WARN: No WER columns found; correlation step will be skipped.")

    speaker_df = aggregate_by_speaker(df, wer_columns)

    print("Segment-level dialect strength summary:")
    print(df["dialect_strength"].describe())

    print("\nSpeaker-level dialect strength summary:")
    print(speaker_df["dialect_strength_mean"].describe())

    if wer_columns:
        print("\nCorrelations (speaker-level):")
        for line in correlation_report(speaker_df, [f"{col}_mean" for col in wer_columns]):
            print("  ", line)

    if not args.no_write:
        if args.segment_output:
            args.segment_output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.segment_output, index=False)
            print(f"Wrote segment-level CSV to {args.segment_output}")
        if args.speaker_output:
            args.speaker_output.parent.mkdir(parents=True, exist_ok=True)
            speaker_df.to_csv(args.speaker_output, index=False)
            print(f"Wrote speaker-level CSV to {args.speaker_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
