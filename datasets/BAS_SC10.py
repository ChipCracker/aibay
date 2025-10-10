from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from paths import BAS_SC10_PATH

_RE_TAG = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!])")
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

_SPEAKER_DIR_PATTERN = re.compile(r"^[a-z]{2}[mw]\d$", re.IGNORECASE)
_WAV_FILENAME_PATTERN = re.compile(
    r"^(?P<speaker>[a-z]{2}[mw]\d)(?P<type>[a-f])(?P<index>\d{3})\.wav$",
    re.IGNORECASE,
)

_RECORDING_TYPE_INFO: Dict[str, Tuple[str, str]] = {
    "a": ("A", "Gelesene, phonetisch ausbalancierte Sätze (Deutsch)"),
    "b": ("B", "Gelesene Zahlenfolgen 1–100 (Deutsch)"),
    "c": ("C", "Gelesene Geschichte »Der Nordwind und die Sonne« (Deutsch)"),
    "d": ("D", "Gelesene Geschichte »Der Nordwind und die Sonne« (L1)"),
    "e": ("E", "Dialog mit deutschsprachiger Bezugsperson"),
    "f": ("F", "Freies Nacherzählen einer Geschichte (Deutsch)"),
}

_SPEAKER_METADATA_KEYS = ("volume", "language_id", "language", "sex", "speaker_number")
_SPEAKER_METADATA_COLUMN_MAP = {
    "volume": "speaker_volume",
    "language_id": "speaker_language_id",
    "language": "speaker_language",
    "sex": "speaker_sex",
    "speaker_number": "speaker_number",
}


def _replace_umlaut(match: re.Match[str]) -> str:
    char = match.group(1)
    return _UMLAUT_MAP.get(char, char)


def _replace_eszett(match: re.Match[str]) -> str:
    return "ß"


def _clean_token(token: str) -> str:
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
    token = _RE_UMLAUT.sub(_replace_umlaut, token)
    token = _RE_ESZETT.sub(_replace_eszett, token)
    token = token.replace('"', "")
    token = token.strip()
    return token


def _clean_free_text(text: str) -> str:
    text = text.replace("*", "")
    text = text.replace("%", "")
    text = _RE_UMLAUT.sub(_replace_umlaut, text)
    text = _RE_ESZETT.sub(_replace_eszett, text)
    text = text.replace("``", '"').replace("''", '"')
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


def _join_tokens(tokens: Iterable[str]) -> Optional[str]:
    cleaned = [token for token in tokens if token]
    if not cleaned:
        return None
    text = " ".join(cleaned)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip() or None


def _read_par_transcriptions(par_path: Path) -> tuple[Optional[str], Optional[str]]:
    if not par_path.is_file():
        return None, None

    ort_tokens: list[str] = []
    tr2_tokens: list[str] = []
    with par_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ORT:\t"):
                parts = line.rstrip("\n").split("\t", maxsplit=2)
                if len(parts) < 3:
                    continue
                token = _clean_token(parts[2])
                if token:
                    ort_tokens.append(token)
            elif line.startswith("TR2:\t"):
                parts = line.rstrip("\n").split("\t", maxsplit=2)
                if len(parts) < 3:
                    continue
                token = _clean_token(parts[2])
                if token:
                    tr2_tokens.append(token)

    return _join_tokens(ort_tokens), _join_tokens(tr2_tokens)


def _parse_trl_file(trl_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with trl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if ":" not in stripped:
                continue
            recording_id, text = stripped.split(":", 1)
            recording_id = recording_id.strip().lower()
            text = _clean_free_text(text)
            if recording_id and text:
                mapping[recording_id] = text
    return mapping


def _load_trl_transcriptions(speaker_dir: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for trl_path in speaker_dir.glob("*.trl"):
        if trl_path.name.startswith("._"):
            continue
        try:
            transcripts.update(_parse_trl_file(trl_path))
        except OSError:
            continue
    return transcripts


def _iter_speaker_directories(dataset_root: Path) -> Iterable[Path]:
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if _SPEAKER_DIR_PATTERN.match(entry.name):
            yield entry


def _load_speaker_metadata(dataset_root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    doc_candidates = [
        dataset_root / "doc" / "SC10_spk.txt",
        dataset_root / "SC10_spk.txt",
        dataset_root / "vdata" / "BAS" / "SC10_2" / "doc" / "SC10_spk.txt",
    ]

    metadata_path = next((path for path in doc_candidates if path.is_file()), None)
    if metadata_path is None:
        return {}

    metadata: Dict[str, Dict[str, Optional[str]]] = {}
    with metadata_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.lower().startswith("volume"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            volume, language_id, sex, number = parts[:4]
            language = " ".join(parts[4:]) if len(parts) > 4 else None
            speaker_id = f"{language_id}{sex}{number}".lower()
            metadata[speaker_id] = {
                "volume": volume,
                "language_id": language_id,
                "sex": sex,
                "speaker_number": number,
                "language": language,
            }
    return metadata


def load_sc10_dataframe(
    dataset_root: Path | str | None = None,
    *,
    include_missing_transcriptions: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame with BAS SC10 audio paths and transcriptions.

    Parameters
    ----------
    dataset_root:
        Base directory containing the BAS SC10 corpus. Defaults to the value
        configured in :mod:`paths`.
    include_missing_transcriptions:
        When ``True`` (default), rows without an available transcription are
        still returned. Set to ``False`` to drop those rows.
    """
    root = Path(dataset_root or BAS_SC10_PATH)
    if not root.exists():
        raise FileNotFoundError(
            "BAS SC10 dataset path not found. Provide dataset_root or set the "
            "DATASETS_PATH environment variable."
        )

    speaker_metadata = _load_speaker_metadata(root)
    rows: list[dict[str, object]] = []

    for speaker_dir in _iter_speaker_directories(root):
        speaker_id = speaker_dir.name.lower()
        metadata_entry = speaker_metadata.get(speaker_id, {})
        trl_lookup = _load_trl_transcriptions(speaker_dir)

        for wav_path in sorted(speaker_dir.glob("*.wav")):
            if wav_path.name.startswith("._"):
                continue
            match = _WAV_FILENAME_PATTERN.match(wav_path.name)
            if not match:
                continue

            recording_base = match.group(0)[:-4].lower()
            type_letter = match.group("type").lower()
            utterance_index = int(match.group("index"))

            par_path = wav_path.with_suffix(".par")
            ort_text, tr2_text = _read_par_transcriptions(par_path)

            if not ort_text:
                ort_text = trl_lookup.get(recording_base)

            if not ort_text and not include_missing_transcriptions:
                continue

            record_type_id, record_type_desc = _RECORDING_TYPE_INFO.get(
                type_letter, (type_letter.upper(), "Unbekannter Aufnahmetyp")
            )

            row: dict[str, object] = {
                "speaker_id": speaker_id,
                "recording_id": recording_base,
                "recording_type_id": record_type_id,
                "recording_type": record_type_desc,
                "utterance_index": utterance_index,
                "audio_path": str(wav_path),
                "gt_transcription": ort_text,
                "dialect_gt_transcription": tr2_text,
            }

            for key in _SPEAKER_METADATA_KEYS:
                row_key = _SPEAKER_METADATA_COLUMN_MAP[key]
                row[row_key] = metadata_entry.get(key)

            rows.append(row)

    columns = [
        "speaker_id",
        "recording_id",
        "recording_type_id",
        "recording_type",
        "utterance_index",
        "audio_path",
        "gt_transcription",
        "dialect_gt_transcription",
        "speaker_volume",
        "speaker_language_id",
        "speaker_language",
        "speaker_sex",
        "speaker_number",
    ]

    df = pd.DataFrame(rows)
    return df.reindex(columns=columns)


__all__ = ["load_sc10_dataframe"]
