from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class InMemoryAudio:
    """Container for in-memory mono audio suitable for ASR pipelines."""

    samples: np.ndarray
    sample_rate: int
    original_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            self.samples = self.samples.reshape(-1)
        if self.samples.dtype != np.float32:
            self.samples = self.samples.astype(np.float32)

    def to_whisper_input(self) -> dict[str, object]:
        """Return input compatible with Hugging Face ASR pipelines."""
        return {
            "array": self.samples,
            "sampling_rate": self.sample_rate,
        }

    def to_wav_bytes(self) -> bytes:
        """Return the audio encoded as a 16-bit PCM WAV byte string."""
        clipped = np.clip(self.samples, -1.0, 1.0)
        int16_samples = (clipped * 32767.0).astype("<i2")

        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(int16_samples.tobytes())
            return buffer.getvalue()

    def descriptor(self) -> str:
        """Human-friendly source identifier used for logging."""
        return self.original_path or "<memory>"


__all__ = ["InMemoryAudio"]
