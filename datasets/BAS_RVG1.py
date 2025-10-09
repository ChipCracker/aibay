import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from paths import BAS_RVG1_PATH

_RE_REPEAT = re.compile(r"\+/(.+?)/\+")
_RE_SENTENCE_BREAK = re.compile(r"-/(.+?)/-")
_RE_VARIANT = re.compile(r"<!(?:\w+)\s+([^>]+)>")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_UMlaut = re.compile(r'"([AOUaou])')
_RE_ESZETT = re.compile(r'"([sS])')
_RE_WHITESPACE = re.compile(r"\s+")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!])")
_RE_BRACKET_CONTENT = re.compile(r"<([^>]+)>")

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


def _clean_trl_text(raw_text: str) -> str:
    """Convert a raw TRL transliteration string into plain text."""
    text = _RE_REPEAT.sub(r"\1", raw_text)
    text = _RE_SENTENCE_BREAK.sub(r"\1", text)
    text = _RE_VARIANT.sub(r"\1", text)
    text = _RE_TAG.sub(" ", text)
    text = text.replace("*", "")
    text = text.replace("%", "")

    text = _RE_UMlaut.sub(_replace_umlaut, text)
    text = _RE_ESZETT.sub(_replace_eszett, text)
    text = text.replace("``", '"').replace("''", '"')
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


def _read_trl_transcription(trl_path: Path) -> Optional[str]:
    """Read and clean the transcription from a TRL file."""
    if not trl_path.is_file():
        return None

    lines: List[str] = []
    with trl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            lines.append(stripped)

    if not lines:
        return None

    joined = " ".join(lines)
    if ":" in joined:
        _, joined = joined.split(":", 1)
    return _clean_trl_text(joined)


def _clean_ort_token(token: str) -> str:
    """Normalise a single ORT token from a Partitur file."""
    token = token.strip()
    if not token or token == "#":
        return ""

    if token.startswith("<") and token.endswith(">"):
        return ""

    token = token.replace("*", "")
    token = token.replace("=", "")
    token = token.replace("#", "")
    token = token.replace(":>", "")
    token = token.replace("_", " ")
    token = token.replace("``", '"').replace("''", '"')
    token = _RE_TAG.sub(" ", token)
    token = token.replace(":>", "")
    token = _RE_UMlaut.sub(_replace_umlaut, token)
    token = _RE_ESZETT.sub(_replace_eszett, token)
    token = token.replace('"', "")
    token = token.strip()
    return token


def _clean_tr2_token(token: str) -> str:
    """Normalise a single TR2 token from a Partitur file."""
    token = token.strip()
    if not token:
        return ""

    token_stripped = token.strip()
    if token_stripped.startswith("+/") and token_stripped.endswith("/+"):
        return ""
    if token_stripped.startswith("-/") and token_stripped.endswith("/-"):
        return ""

    selected: Optional[str] = None
    variant_matches: List[str] = []
    non_variant_matches: List[str] = []

    for match in _RE_BRACKET_CONTENT.finditer(token):
        content = match.group(1).strip()
        if not content:
            continue
        if "#" in content:
            continue
        if content.startswith(":") or content.startswith(";"):
            continue
        if content.startswith("!"):
            stripped = re.sub(r"^!+\d*\s*", "", content).strip()
            if stripped:
                variant_matches.append(stripped)
            continue

        non_variant_matches.append(content)

    if variant_matches:
        selected = variant_matches[-1]
    else:
        base_candidate = re.sub(r"<[^>]*>", " ", token)
        if any(ch.isalpha() for ch in base_candidate):
            selected = base_candidate
        else:
            for candidate in non_variant_matches:
                candidate = candidate.lstrip("~")
                candidate = candidate.strip()
                if not candidate:
                    continue
                if len(candidate) == 1 and candidate.upper() in {"A", "P", "Z"}:
                    continue
                selected = candidate
                break

    value = selected if selected else token

    value = value.replace(r"\'", "'")
    value = value.replace("~", "")
    value = value.replace("+/", "")
    value = value.replace("/+", "")
    value = value.replace("*", "")
    value = value.replace("=", "")
    value = value.replace("$", "")
    value = value.replace(":>", "")
    value = value.replace("#", "")
    value = value.replace("_", " ")
    value = value.replace("``", '"').replace("''", '"')
    value = _RE_TAG.sub(" ", value)
    value = re.sub(r"<\s*>", " ", value)
    value = _RE_UMlaut.sub(_replace_umlaut, value)
    value = _RE_ESZETT.sub(_replace_eszett, value)
    value = value.replace('"', "")
    value = value.replace("%", "")
    value = _RE_WHITESPACE.sub(" ", value).strip()
    value = value.replace(" ", "")
    return value


def _read_par_transcriptions(par_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract ORT (high German) and TR2 (dialect) transcriptions."""
    if not par_path.is_file():
        return None, None

    tokens: List[str] = []
    tr2_tokens: List[str] = []
    with par_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("ORT:\t"):
                parts = line.rstrip("\n").split("\t", maxsplit=2)
                if len(parts) < 3:
                    continue
                token = _clean_ort_token(parts[2])
                if token:
                    tokens.append(token)
            elif line.startswith("TR2:\t"):
                parts = line.rstrip("\n").split("\t", maxsplit=2)
                if len(parts) < 3:
                    continue
                token = _clean_tr2_token(parts[2])
                if token:
                    tr2_tokens.append(token)

    ort_text = ""
    if tokens:
        ort_text = " ".join(tokens)
        ort_text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", ort_text)
        ort_text = _RE_WHITESPACE.sub(" ", ort_text)
        ort_text = ort_text.strip()

    tr2_text = ""
    if tr2_tokens:
        raw_tr2 = " ".join(tr2_tokens)
        tr2_text = _clean_trl_text(raw_tr2)

    return (ort_text or None, tr2_text or None)


def _iter_speaker_directories(dataset_root: Path) -> Iterable[Path]:
    """Yield valid speaker directories within the dataset root."""
    for entry in sorted(dataset_root.iterdir()):
        if entry.is_dir() and entry.name.isdigit():
            yield entry


def _resolve_audio_path(speaker_dir: Path, channel: str) -> Optional[Path]:
    """Find the wav file for the requested microphone channel."""
    pattern = f"sp1{channel.lower()}*.wav"
    candidates = sorted(speaker_dir.glob(pattern))
    if candidates:
        return candidates[0]

    fallback_pattern = f"sp1{channel.lower()}*.nis"
    nis_candidates = sorted(speaker_dir.glob(fallback_pattern))
    if nis_candidates:
        return nis_candidates[0]
    return None


def load_sp1_dataframe(
    dataset_root: Path | str | None = None,
    *,
    channel: str = "a",
) -> pd.DataFrame:
    """Return a DataFrame with SP1 audio paths and ground truth transcriptions.

    Parameters
    ----------
    dataset_root:
        Base directory containing the ``RVG1_CLARIN`` corpus. If omitted, the
        value from :mod:`paths` is used.
    channel:
        Microphone channel identifier to use for the audio files. Channel ``c``
        corresponds to the Sennheiser HD/MD 410 headset recordings.

    Returns
    -------
    pandas.DataFrame
        Columns: ``speaker_id``, ``audio_path``, ``transcription``,
        ``dialect_gt_transcription``.
    """
    if channel.lower() not in {"a", "b", "c", "d"}:
        raise ValueError("channel must be one of 'a', 'b', 'c', or 'd'")

    root = Path(dataset_root or BAS_RVG1_PATH)
    if not root or not root.exists():
        raise FileNotFoundError(
            "BAS RVG1 dataset path not found. Provide dataset_root or set the "
            "DATASETS_PATH environment variable."
        )

    rows: List[Tuple[str, str, str, Optional[str]]] = []
    for speaker_dir in _iter_speaker_directories(root):
        speaker_id = speaker_dir.name
        audio_path = _resolve_audio_path(speaker_dir, channel)
        if not audio_path:
            continue

        par_path = audio_path.with_suffix(".par")
        transcription, dialect_transcription = _read_par_transcriptions(par_path)
        if not transcription:
            trl_path = speaker_dir / f"sp1{speaker_id}.trl"
            transcription = _read_trl_transcription(trl_path)

        if not transcription:
            continue

        rows.append(
            (
                speaker_id,
                str(audio_path),
                transcription,
                dialect_transcription,
            )
        )

    return pd.DataFrame(
        rows,
        columns=["speaker_id", "audio_path", "transcription", "dialect_gt_transcription"],
    )


__all__ = ["load_sp1_dataframe"]
