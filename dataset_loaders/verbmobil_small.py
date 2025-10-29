from __future__ import annotations

import logging
import re
import wave
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from audio_utils import InMemoryAudio
from paths import (
    VERBMOBIL_SMALL_AUDIO_PATH,
    VERBMOBIL_SMALL_PATH,
    VERBMOBIL_SMALL_TRL_PATH,
)

_LOGGER = logging.getLogger(__name__)

_RE_UMLAUT = re.compile(r'"([AOUaou])')
_RE_ESZETT = re.compile(r'"([sS])')
_RE_TAG = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_TRAILING_PUNCT = re.compile(r"[;:.]+$")
_RE_TURN_INFO = re.compile(r"^([A-ZÄÖÜß]+)(\d+)?([A-Z]?)$")

_UMLAUT_MAP = {
    "a": "ä",
    "A": "Ä",
    "o": "ö",
    "O": "Ö",
    "u": "ü",
    "U": "Ü",
}

_RAW_SUFFIXES: tuple[str, ...] = ("", ".ssg", ".SSG", ".pcm", ".PCM", ".raw", ".RAW")
_WAV_SUFFIXES: tuple[str, ...] = (".wav", ".WAV")


def _replace_umlaut(match: re.Match[str]) -> str:
    char = match.group(1)
    return _UMLAUT_MAP.get(char, char)


def _replace_eszett(_: re.Match[str]) -> str:
    return "ß"


def _clean_transcription(raw_text: str) -> str:
    """Normalise a raw TRL transcription line."""
    text = raw_text.strip()
    if not text:
        return ""

    text = _RE_TRAILING_PUNCT.sub("", text)
    text = _RE_TAG.sub(" ", text)
    text = text.replace("_", " ")
    text = _RE_UMLAUT.sub(_replace_umlaut, text)
    text = _RE_ESZETT.sub(_replace_eszett, text)
    text = text.replace('"', "")
    text = text.replace("``", '"').replace("''", '"')
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


def _build_audio_candidates(audio_root: Path, utterance_id: str) -> Iterable[Path]:
    """Yield plausible file paths for a given utterance identifier."""
    base_names = {utterance_id, utterance_id.replace(":", "_"), utterance_id.replace(":", "-")}

    for base_name in base_names:
        direct = audio_root / base_name
        yield direct
        for suffix in (*_RAW_SUFFIXES, *_WAV_SUFFIXES):
            if not suffix:
                continue
            yield audio_root / f"{base_name}{suffix}"


def _resolve_audio_path(audio_root: Path, utterance_id: str) -> Optional[Path]:
    """Locate the audio file corresponding to ``utterance_id``."""
    for candidate in _build_audio_candidates(audio_root, utterance_id):
        if candidate.is_file():
            return candidate
    return None


def _parse_identifier(utterance_id: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract conversation, speaker code, turn index, and channel from an utterance identifier."""
    conversation_id: Optional[str] = None
    speaker_code: Optional[str] = None
    turn_index: Optional[str] = None
    channel: Optional[str] = None

    if ":" in utterance_id:
        conversation_id, remainder = utterance_id.split(":", 1)
        remainder = remainder.strip()
        match = _RE_TURN_INFO.match(remainder)
        if match:
            speaker_code = match.group(1) or None
            turn_index = match.group(2) or None
            channel = match.group(3) or None
        elif remainder:
            speaker_code = remainder
    else:
        if "." in utterance_id:
            conversation_id = utterance_id.split(".", 1)[0]
        else:
            conversation_id = utterance_id

    return conversation_id, speaker_code, turn_index, channel


def _load_audio_into_memory(audio_path: Path, *, sample_rate: int) -> InMemoryAudio:
    """Return an in-memory audio segment from a PCM or WAV file."""
    suffix_lower = audio_path.suffix.lower()
    if suffix_lower in {ext.lower() for ext in _WAV_SUFFIXES}:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            if channels != 1:
                _LOGGER.debug(
                    "Downmixing %s with %d channels to mono for ASR compatibility.",
                    audio_path,
                    channels,
                )
            frames = wav_file.readframes(wav_file.getnframes())
            wav_sample_rate = wav_file.getframerate()

        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
        if wav_sample_rate != sample_rate:
            _LOGGER.debug(
                "Keeping native sample rate %d Hz for %s (default loader sample rate %d Hz).",
                wav_sample_rate,
                audio_path,
                sample_rate,
            )
        return InMemoryAudio(samples=samples, sample_rate=wav_sample_rate, original_path=str(audio_path))

    raw_bytes = audio_path.read_bytes()
    if not raw_bytes:
        raise ValueError(f"Empty audio file encountered: {audio_path}")

    samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32767.0
    return InMemoryAudio(samples=samples, sample_rate=sample_rate, original_path=str(audio_path))


def load_verbmobil_small_dataframe(
    dataset_root: Path | str | None = None,
    *,
    audio_dir: Path | str | None = None,
    trl_path: Path | str | None = None,
    sample_rate: int = 16_000,
    convert_raw_audio: bool = True,
) -> pd.DataFrame:
    """Load the Verbmobil small corpus as a DataFrame ready for ASR evaluation.

    Parameters
    ----------
    dataset_root:
        Optional override for the corpus location. Defaults to ``VERBMOBIL_SMALL_PATH``.
    audio_dir:
        Optional override for the directory containing the SSG/PCM audio files.
        Defaults to ``VERBMOBIL_SMALL_AUDIO_PATH`` or the ``SSG`` directory beneath
        ``dataset_root`` / ``VERBMOBIL_SMALL_PATH``.
    trl_path:
        Optional override for the TRL transcript index file. Defaults to
        ``VERBMOBIL_SMALL_TRL_PATH`` or ``verbmobil_small.trl`` beneath the dataset root.
    sample_rate:
        Sampling rate used when interpreting raw PCM segments (default: 16 kHz).
    convert_raw_audio:
        When True, raw PCM segments are materialised as in-memory audio objects.

    Returns
    -------
    pd.DataFrame
        Columns include ``utterance_id``, ``conversation_id``, ``speaker_code``,
        ``turn_index``, ``channel``, ``audio_source``, ``audio_path``,
        ``gt_transcription``, and ``gt_transcription_raw``.
    """
    root = Path(dataset_root) if dataset_root else Path(VERBMOBIL_SMALL_PATH) if VERBMOBIL_SMALL_PATH else None

    resolved_trl_path: Optional[Path]
    if trl_path:
        resolved_trl_path = Path(trl_path)
    elif VERBMOBIL_SMALL_TRL_PATH:
        resolved_trl_path = Path(VERBMOBIL_SMALL_TRL_PATH)
    elif root is not None:
        resolved_trl_path = root / "verbmobil_small.trl"
    else:
        resolved_trl_path = None

    if not resolved_trl_path or not resolved_trl_path.is_file():
        raise FileNotFoundError(
            "Verbmobil TRL file not found. Provide trl_path or set VERBMOBIL_SMALL_TRL_FILE_PATH."
        )

    resolved_audio_root: Optional[Path]
    if audio_dir:
        resolved_audio_root = Path(audio_dir)
    elif VERBMOBIL_SMALL_AUDIO_PATH:
        resolved_audio_root = Path(VERBMOBIL_SMALL_AUDIO_PATH)
    elif root is not None:
        resolved_audio_root = root / "SSG"
    else:
        resolved_audio_root = None

    if not resolved_audio_root or not resolved_audio_root.is_dir():
        raise FileNotFoundError(
            "Verbmobil audio directory not found. Provide audio_dir or set VERBMOBIL_SMALL_AUDIO_PATH."
        )

    rows: list[dict[str, object]] = []
    missing_audio: list[str] = []

    with resolved_trl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "\t" not in line:
                continue

            utterance_id, transcript_raw = line.split("\t", 1)
            transcript_raw = transcript_raw.strip()
            cleaned_transcription = _clean_transcription(transcript_raw)
            original_transcript = _RE_TRAILING_PUNCT.sub("", transcript_raw.strip())

            audio_path = _resolve_audio_path(resolved_audio_root, utterance_id)
            if not audio_path or not audio_path.is_file():
                missing_audio.append(utterance_id)
                continue

            audio_source: Optional[InMemoryAudio] = None
            if convert_raw_audio:
                try:
                    audio_source = _load_audio_into_memory(audio_path, sample_rate=sample_rate)
                except Exception as exc:
                    _LOGGER.warning("Failed to load %s into memory: %s", audio_path, exc)

            conversation_id, speaker_code, turn_index, channel = _parse_identifier(utterance_id)

            rows.append(
                {
                    "utterance_id": utterance_id,
                    "conversation_id": conversation_id,
                    "speaker_code": speaker_code,
                    "turn_index": turn_index,
                    "channel": channel,
                    "audio_source": audio_source,
                    "audio_path": str(audio_path),
                    "gt_transcription": cleaned_transcription,
                    "gt_transcription_raw": original_transcript,
                }
            )

    if missing_audio:
        sample_ids = ", ".join(missing_audio[:5])
        _LOGGER.warning(
            "Skipped %d Verbmobil utterances without audio files (examples: %s)%s",
            len(missing_audio),
            sample_ids,
            " ..." if len(missing_audio) > 5 else "",
        )

    if not rows:
        raise ValueError("No Verbmobil_small utterances with audio were found. Verify the dataset paths.")

    columns = [
        "utterance_id",
        "conversation_id",
        "speaker_code",
        "turn_index",
        "channel",
        "audio_source",
        "audio_path",
        "gt_transcription",
        "gt_transcription_raw",
    ]
    return pd.DataFrame(rows, columns=columns)


__all__ = ["load_verbmobil_small_dataframe"]
