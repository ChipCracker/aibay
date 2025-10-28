from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from paths import SWITCHBOARD_BENCHMARK_PATH

_LOGGER = logging.getLogger(__name__)

_COMMENT_PREFIX = ";;"
_STM_FILENAME = "switchboard-benchmark.stm"
_FILLER_PREFIX = "%"


def _clean_token(token: str) -> str:
    """Return a normalised token suitable for WER comparison."""
    value = token.strip()
    if not value:
        return ""

    # Remove surrounding parentheses that mark disfluencies/alternations
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()

    if not value:
        return ""

    # Drop filler markers such as %HESITATION
    if value.startswith(_FILLER_PREFIX):
        return ""

    # Resolve alternations by keeping the last option (usually the corrected form)
    if "/" in value:
        alternatives = [alt for alt in value.split("/") if alt]
        if alternatives:
            value = alternatives[-1]
        else:
            return ""

    # Strip leading/trailing disfluency hyphens, e.g. -T'S, I-
    value = value.strip("-")
    if not value:
        return ""

    # Remove residual hyphen markers inside the token so UH-HUH → UHHUH
    value = value.replace("-", "")
    if not value:
        return ""

    # Collapse repeated apostrophes (rare artefact)
    while "''" in value:
        value = value.replace("''", "'")

    return value.lower()


def _normalise_transcript(text: str) -> str:
    tokens = [_clean_token(token) for token in text.split()]
    return " ".join(token for token in tokens if token)


def _parse_stm_file(stm_path: Path) -> List[dict[str, object]]:
    segments: List[dict[str, object]] = []

    with stm_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(_COMMENT_PREFIX):
                continue

            parts = line.split(None, 6)
            if len(parts) < 7:
                _LOGGER.warning("Skipping malformed STM line: %s", raw_line.rstrip())
                continue

            conversation_id, channel, file_id, start, end, label, transcript = parts
            transcript = transcript.strip()

            segments.append(
                {
                    "conversation_id": conversation_id,
                    "channel": channel,
                    "file_id": file_id,
                    "start": float(start),
                    "end": float(end),
                    "label": label,
                    "transcript_raw": transcript,
                    "transcript_clean": _normalise_transcript(transcript),
                }
            )

    return segments


def load_switchboard_benchmark_dataframe(dataset_root: Path | str | None = None) -> pd.DataFrame:
    """Return a DataFrame for the Mod9 Switchboard benchmark dataset.

    The resulting frame contains one row per conversation channel with the
    concatenated ground-truth transcription.
    """
    root = Path(dataset_root or SWITCHBOARD_BENCHMARK_PATH)
    if not root.is_dir():
        raise FileNotFoundError(
            "Switchboard benchmark dataset path not found. Provide dataset_root or "
            "set DATASETS_PATH / DATASETS_ROOT environment variables."
        )

    stm_path = root / _STM_FILENAME
    if not stm_path.is_file():
        raise FileNotFoundError(f"STM file not found: {stm_path}")

    segments = _parse_stm_file(stm_path)
    if not segments:
        raise ValueError(f"No segments parsed from STM file: {stm_path}")

    segment_collector: Dict[tuple[str, str], List[dict[str, object]]] = defaultdict(list)

    for segment in segments:
        key = (segment["conversation_id"], segment["channel"])
        segment_collector[key].append(segment)

    rows: List[dict[str, object]] = []
    for (conversation_id, channel), items in sorted(segment_collector.items()):
        items = sorted(items, key=lambda segment: segment["start"])
        file_ids = {segment["file_id"] for segment in items}
        if len(file_ids) != 1:
            _LOGGER.warning(
                "Multiple file IDs for conversation %s channel %s: %s",
                conversation_id,
                channel,
                sorted(file_ids),
            )

        file_id = sorted(file_ids)[0]
        audio_path = root / f"{file_id}.wav"
        if not audio_path.is_file():
            _LOGGER.warning("Audio file missing for %s: %s", file_id, audio_path)

        raw_segments = [segment["transcript_raw"] for segment in items if segment["transcript_raw"]]
        clean_segments = [segment["transcript_clean"] for segment in items if segment["transcript_clean"]]

        segment_payload = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "label": segment["label"],
                "transcript_raw": segment["transcript_raw"],
                "transcript_clean": segment["transcript_clean"],
            }
            for segment in items
        ]

        rows.append(
            {
                "conversation_id": conversation_id,
                "channel": channel,
                "speaker_id": f"{conversation_id}_{channel}",
                "audio_path": str(audio_path),
                "segment_count": len(items),
                "duration_s": max(segment["end"] for segment in items),
                "gt_transcription": " ".join(clean_segments).strip(),
                "gt_transcription_raw": " ".join(raw_segments).strip(),
                "segments": segment_payload,
            }
        )

    columns = [
        "conversation_id",
        "channel",
        "speaker_id",
        "audio_path",
        "segment_count",
        "duration_s",
        "gt_transcription",
        "gt_transcription_raw",
        "segments",
    ]

    return pd.DataFrame(rows, columns=columns)


__all__ = ["load_switchboard_benchmark_dataframe"]
