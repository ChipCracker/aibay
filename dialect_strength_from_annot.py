#!/usr/bin/env python3
"""Compute dialect strength from BAS RVG1 annotation JSON files.

For every *.annot.json file the script compares the token in the ORT (reference)
layer with the corresponding TR2 (dialect) layer. A token is counted as
“dialektisch ersetzt”, when at least one of the following holds:

* TR2 contains explicit dialect markup (e.g. ``<!...>`` or ``+/.../+`` or ``#``).
* The cleaned TR2 token differs from the cleaned ORT token.

Dialect strength for a recording is the fraction of replaced tokens.
Speaker-level scores are aggregated (mean/median/max/share>0).

If an additional WER CSV is provided the script joins the speaker statistics and
prints Pearson/Spearman correlations between dialect strength and WER.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from paths import BAS_RVG1_PATH, OUTPUT_PATH

# Regex patterns mirroring the cleaning logic from dataset_loaders.BAS_RVG1
_RE_TAG = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_LEADING_PUNCT = re.compile(r"^[^\wäöüß']+")
_RE_TRAILING_PUNCT = re.compile(r"[^\wäöüß']+$")
_RE_UMLAUT = re.compile(r'"([AOUaou])')
_RE_ESZETT = re.compile(r'"([sS])')

_UMLAUT_MAP = {
    "a": "ä",
    "A": "Ä",
    "o": "ö",
    "O": "Ö",
    "u": "ü",
    "U": "Ü",
}


def _replace_umlaut(match: re.Match[str]) -> str:
    char = match.group(1)
    return _UMLAUT_MAP.get(char, char)


def _replace_eszett(match: re.Match[str]) -> str:
    return "ß"


def normalise_token(text: str | None) -> str:
    """Return a lower-case token stripped from markup & punctuation."""
    if not text:
        return ""

    value = text.strip()
    if not value:
        return ""

    value = value.replace("``", '"').replace("''", '"')
    value = value.replace("*", "").replace("=", "").replace("~", "")
    value = _RE_UMLAUT.sub(_replace_umlaut, value)
    value = _RE_ESZETT.sub(_replace_eszett, value)
    value = value.replace('"', "")
    value = _RE_TAG.sub(" ", value)
    value = value.replace(".", " ").replace(",", " ")
    value = _RE_WHITESPACE.sub(" ", value).strip()
    value = _RE_LEADING_PUNCT.sub("", value)
    value = _RE_TRAILING_PUNCT.sub("", value)
    return value.lower()


def tr2_has_dialect_markup(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if "<!" in stripped or "+/" in stripped or stripped.startswith("#"):
        return True
    return False


def token_replaced(ort_raw: str | None, tr2_raw: str | None) -> tuple[bool, str, str]:
    """Return (is_replaced, ort_token, tr2_token)."""
    ort_token = normalise_token(ort_raw)
    if not ort_token:
        return False, "", ""

    if not tr2_raw or not tr2_raw.strip():
        return False, ort_token, ""

    if tr2_has_dialect_markup(tr2_raw):
        return True, ort_token, normalise_token(tr2_raw)

    tr2_token = normalise_token(tr2_raw)
    if not tr2_token:
        return False, ort_token, ""

    return ort_token != tr2_token, ort_token, tr2_token


def _label_lookup(labels: list[dict], name: str) -> str | None:
    for entry in labels:
        if entry.get("name") == name:
            return entry.get("value")
    return None


@dataclass
class RecordingStats:
    path: Path
    recording_id: str
    speaker_id: str
    tokens_total: int
    tokens_replaced: int
    dialect_strength: float


def parse_annotation(path: Path) -> RecordingStats | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    recording_id = data.get("name", path.stem)

    # Sprecher-ID aus Bundle-Level (Label "SPN")
    speaker_id: str | None = None
    for level in data.get("levels", []):
        if level.get("name") == "bundle":
            for item in level.get("items", []):
                spn = _label_lookup(item.get("labels", []), "SPN")
                if spn:
                    speaker_id = spn.zfill(3)
                    break
        if speaker_id:
            break
    if not speaker_id:
        parent_name = path.parent.name
        if parent_name.isdigit():
            speaker_id = parent_name
        else:
            # Fallback: letzte drei Ziffern des Recording-Namens
            speaker_id = recording_id[-3:]

    ort_level = next((lvl for lvl in data.get("levels", []) if lvl.get("name") == "ORT"), None)
    if not ort_level:
        return None

    total = 0
    replaced = 0
    for item in ort_level.get("items", []):
        ort_raw = _label_lookup(item.get("labels", []), "ORT")
        tr2_raw = _label_lookup(item.get("labels", []), "TR2")
        is_replaced, ort_token, _ = token_replaced(ort_raw, tr2_raw)
        if not ort_token:
            continue
        total += 1
        if is_replaced:
            replaced += 1

    strength = (replaced / total) if total else 0.0
    return RecordingStats(
        path=path,
        recording_id=recording_id,
        speaker_id=speaker_id,
        tokens_total=total,
        tokens_replaced=replaced,
        dialect_strength=strength,
    )


def aggregate_speaker_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("speaker_id").agg(
        dialect_strength_mean=("dialect_strength", "mean"),
        dialect_strength_median=("dialect_strength", "median"),
        dialect_strength_max=("dialect_strength", "max"),
        tokens_total=("tokens_total", "sum"),
        tokens_replaced=("tokens_replaced", "sum"),
    )
    grouped["dialect_strength_share_nonzero"] = (
        df.assign(flag=df["dialect_strength"] > 0)
        .groupby("speaker_id")["flag"]
        .mean()
    )
    return grouped.reset_index()


def aggregate_wer(wer_df: pd.DataFrame) -> pd.DataFrame:
    wer_columns = [col for col in wer_df.columns if col.endswith("_wer")]
    if not wer_columns:
        return pd.DataFrame()
    grouped = wer_df.groupby("speaker_id")[wer_columns].agg(["mean", "median"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    return grouped.reset_index()


def correlations(strength_df: pd.DataFrame, wer_df: pd.DataFrame) -> list[str]:
    if strength_df.empty or wer_df.empty:
        return []

    merged = strength_df.merge(wer_df, on="speaker_id", how="inner")
    merged = merged.dropna(subset=["dialect_strength_mean"])

    lines: list[str] = []
    for col in wer_df.columns:
        if col == "speaker_id":
            continue
        if not col.endswith(("_mean", "_median")):
            continue
        if merged[col].isna().all():
            continue

        pearson = merged["dialect_strength_mean"].corr(merged[col], method="pearson")
        spearman = merged["dialect_strength_mean"].corr(merged[col], method="spearman")
        n = merged[[ "dialect_strength_mean", col]].dropna().shape[0]
        lines.append(f"{col}: pearson={pearson:.4f} spearman={spearman:.4f} (n={n})")
    return lines


def parse_args() -> argparse.Namespace:
    default_root = Path(BAS_RVG1_PATH) if BAS_RVG1_PATH else None
    default_output_dir = Path(OUTPUT_PATH) if OUTPUT_PATH else Path("outputs")

    parser = argparse.ArgumentParser(
        description="Compute dialect strength from BAS RVG1 annotation JSON files."
    )
    parser.add_argument(
        "annot_root",
        type=Path,
        nargs="?" if default_root else None,
        default=default_root,
        help=(
            "Directory containing *.annot.json files. "
            f"Default: {default_root}" if default_root else "Directory containing *.annot.json files."
        ),
    )
    parser.add_argument(
        "--wer-csv",
        type=Path,
        default=None,
        help="Optional CSV with WER results (must contain speaker_id and *_wer columns).",
    )
    parser.add_argument(
        "--recording-output",
        type=Path,
        help="Optional CSV to write recording-level dialect strengths.",
    )
    parser.add_argument(
        "--speaker-output",
        type=Path,
        help="Optional CSV to write speaker-level dialect strengths.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing CSV outputs even if output paths are supplied.",
    )
    args = parser.parse_args()
    if args.annot_root is None:
        parser.error("annot_root is required because DATASETS_PATH/BAS-RVG1 is not configured.")

    if args.recording_output and not args.recording_output.is_absolute():
        args.recording_output = default_output_dir / args.recording_output
    if args.speaker_output and not args.speaker_output.is_absolute():
        args.speaker_output = default_output_dir / args.speaker_output

    return args


def main() -> int:
    args = parse_args()

    print(args.annot_root)
    if not args.annot_root.exists():
        raise FileNotFoundError(f"Annotation root not found: {args.annot_root}")

    annot_files = sorted(args.annot_root.rglob("*.annot.json"))
    if not annot_files:
        raise FileNotFoundError("No *.annot.json files found under the provided root.")

    records: list[RecordingStats] = []
    for path in annot_files:
        stats = parse_annotation(path)
        if stats is None:
            continue
        records.append(stats)

    if not records:
        raise RuntimeError("No usable annotation files parsed.")

    rec_df = pd.DataFrame(
        {
            "recording_id": [r.recording_id for r in records],
            "speaker_id": [r.speaker_id for r in records],
            "tokens_total": [r.tokens_total for r in records],
            "tokens_replaced": [r.tokens_replaced for r in records],
            "dialect_strength": [r.dialect_strength for r in records],
            "annot_path": [str(r.path) for r in records],
        }
    )

    speaker_df = aggregate_speaker_stats(rec_df)

    print("Recording-level dialect strength summary:")
    print(rec_df["dialect_strength"].describe())
    print("\nSpeaker-level dialect strength summary:")
    print(speaker_df["dialect_strength_mean"].describe())

    wer_df = pd.DataFrame()
    if args.wer_csv:
        if not args.wer_csv.is_file():
            raise FileNotFoundError(f"WER CSV not found: {args.wer_csv}")
        raw_wer = pd.read_csv(args.wer_csv)
        if "speaker_id" not in raw_wer.columns:
            raise ValueError("WER CSV must contain a 'speaker_id' column.")
        wer_df = aggregate_wer(raw_wer)
        if not wer_df.empty:
            print("\nCorrelations (speaker-level):")
            for line in correlations(speaker_df, wer_df):
                print("  ", line)

    if not args.no_write:
        if args.recording_output:
            args.recording_output.parent.mkdir(parents=True, exist_ok=True)
            rec_df.to_csv(args.recording_output, index=False)
            print(f"Recorded dialect strengths written to {args.recording_output}")
        if args.speaker_output:
            args.speaker_output.parent.mkdir(parents=True, exist_ok=True)
            out_df = speaker_df.copy()
            if not wer_df.empty:
                out_df = out_df.merge(wer_df, on="speaker_id", how="left")
            out_df.to_csv(args.speaker_output, index=False)
            print(f"Speaker-level summary written to {args.speaker_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
