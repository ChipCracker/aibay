from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from paths import BAS_SC1_PATH

_RE_TAG = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!])")
_RE_UMlaut = re.compile(r'"([AOUaou])')
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


def _clean_ort_token(token: str) -> str:
    token = token.strip()
    if not token or token == "#":
        return ""

    if token.startswith("<") and token.endswith(">"):
        return ""

    token = token.replace("*", "")
    token = token.replace("=", "")
    token = token.replace("_", " ")
    token = token.replace("``", '"').replace("''", '"')
    token = _RE_TAG.sub(" ", token)
    token = _RE_UMlaut.sub(_replace_umlaut, token)
    token = _RE_ESZETT.sub(_replace_eszett, token)
    token = token.replace('"', "")
    token = token.strip()
    return token


def _read_par_transcription(par_path: Path) -> Optional[str]:
    if not par_path.is_file():
        return None

    tokens: list[str] = []
    with par_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ORT:\t"):
                continue
            parts = line.rstrip("\n").split("\t", maxsplit=2)
            if len(parts) < 3:
                continue
            token = _clean_ort_token(parts[2])
            if token:
                tokens.append(token)

    if not tokens:
        return None

    text = " ".join(tokens)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


def _iter_recording_directories(dataset_root: Path) -> Iterable[Path]:
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / f"{entry.name}.wav").is_file():
            yield entry


def _load_speaker_metadata(dataset_root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    metadata_path = dataset_root / "doc" / "sc1_spk.txt"
    if not metadata_path.is_file():
        return {}

    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    with metadata_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                continue
            (
                recording_id,
                mother_tongue_iso639_3,
                country,
                country_iso3166_2,
                sex,
                year,
                subcorpus,
            ) = parts[:7]

            metadata[recording_id] = {
                "mother_tongue_iso639_3": mother_tongue_iso639_3 or None,
                "country": country or None,
                "country_iso3166_2": country_iso3166_2 or None,
                "sex": sex or None,
                "recording_year": year or None,
                "subcorpus": subcorpus or None,
            }
    return metadata


def load_sc1_dataframe(
    dataset_root: Path | str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with BAS SC1 audio paths and ground truth transcriptions."""
    root = Path(dataset_root or BAS_SC1_PATH)
    if not root or not root.exists():
        raise FileNotFoundError(
            "BAS SC1 dataset path not found. Provide dataset_root or set the DATASETS_PATH environment variable."
        )

    metadata = _load_speaker_metadata(root)
    rows: list[dict[str, object]] = []

    for recording_dir in _iter_recording_directories(root):
        recording_id = recording_dir.name
        audio_path = recording_dir / f"{recording_id}.wav"
        par_path = recording_dir / f"{recording_id}.par"

        transcription = _read_par_transcription(par_path)
        if not transcription:
            continue

        row: dict[str, object] = {
            "speaker_id": recording_id,
            "audio_path": str(audio_path),
            "gt_transcription": transcription,
            "dialect_gt_transcription": None,
        }

        meta_entry = metadata.get(recording_id, {})
        row.update(
            {
                "mother_tongue_iso639_3": meta_entry.get("mother_tongue_iso639_3"),
                "country": meta_entry.get("country"),
                "country_iso3166_2": meta_entry.get("country_iso3166_2"),
                "sex": meta_entry.get("sex"),
                "recording_year": meta_entry.get("recording_year"),
                "subcorpus": meta_entry.get("subcorpus"),
            }
        )

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            "speaker_id",
            "audio_path",
            "gt_transcription",
            "dialect_gt_transcription",
            "mother_tongue_iso639_3",
            "country",
            "country_iso3166_2",
            "sex",
            "recording_year",
            "subcorpus",
        ],
    )


__all__ = ["load_sc1_dataframe"]
