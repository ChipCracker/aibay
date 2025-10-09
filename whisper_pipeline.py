from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd


_UNSUPPORTED_GENERATE_KWARGS = {"cache_dir"}


def _sanitize_generate_kwargs(kwargs: Optional[dict] = None) -> dict:
    if not kwargs:
        return {}
    return {key: value for key, value in kwargs.items() if key not in _UNSUPPORTED_GENERATE_KWARGS}


def _ensure_package(name: str, install_hint: str) -> None:
    try:
        __import__(name)
    except ImportError as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            f"The '{name}' package is required for this operation. {install_hint}"
        ) from exc


def load_whisper_model(
    model_name: str = "openai/whisper-large-v3",
    *,
    device: Optional[Union[int, str]] = None,
    torch_dtype: Optional["torch.dtype"] = None,
    chunk_length_s: Optional[float] = 30.0,
    stride_length_s: Optional[Union[float, tuple[float, float]]] = 5.0,
    return_timestamps: Union[bool, str] = True,
    ignore_warning: bool = True,
    generate_kwargs: Optional[dict] = None,
) -> "transformers.pipelines.Pipeline":
    """Load the Hugging Face Whisper pipeline configured for long-form transcription.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier, e.g. ``openai/whisper-large-v3``.
    device:
        Torch device identifier. ``0`` selects the first GPU; ``"cpu"`` forces CPU.
        When omitted, CUDA is used if available.
    torch_dtype:
        Optional dtype override (defaults to ``float16`` on GPU, ``float32`` on CPU).
    chunk_length_s:
        Chunk size in seconds, mirroring the Hugging Face documentation for long-form decoding.
    stride_length_s:
        Stride (overlap) applied between chunks. Can be a single float or a ``(left, right)`` tuple.
    return_timestamps:
        Whether to request timestamp-aware decoding. The Hugging Face docs enable this for long-form audio.
    ignore_warning:
        Forwarded to the pipeline to silence experimental chunking warnings when desired.
    generate_kwargs:
        Default ``generate_kwargs`` applied to every inference call (e.g. language/task).
    """
    _ensure_package("transformers", "Install it via `pip install transformers[audio]`.")
    _ensure_package("torch", "Install PyTorch compatible with your platform.")

    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    resolved_dtype = torch_dtype
    if resolved_dtype is None:
        resolved_dtype = torch.float16 if device != "cpu" else torch.float32

    pipeline_kwargs: dict[str, object] = {
        "model": model_name,
        "device": device,
        "torch_dtype": resolved_dtype,
    }

    if chunk_length_s is not None:
        pipeline_kwargs["chunk_length_s"] = chunk_length_s
    if stride_length_s is not None:
        pipeline_kwargs["stride_length_s"] = stride_length_s
    if return_timestamps is not None:
        pipeline_kwargs["return_timestamps"] = return_timestamps
    if ignore_warning is not None:
        pipeline_kwargs["ignore_warning"] = ignore_warning

    pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)

    if generate_kwargs:
        existing_kwargs = _sanitize_generate_kwargs(getattr(pipe, "generate_kwargs", {}))
        pipe.generate_kwargs = {**existing_kwargs, **_sanitize_generate_kwargs(generate_kwargs)}

    return pipe


def _iter_audio_paths(paths: Iterable[str | Path], show_progress: bool) -> Iterable[Path]:
    iterator = [Path(p) for p in paths]
    if show_progress:
        try:  # pragma: no cover - optional dependency
            from tqdm import tqdm

            return (Path(p) for p in tqdm(iterator, desc="Transcribing", unit="file"))
        except ImportError:
            pass
    return (Path(p) for p in iterator)


def transcribe_dataframe(
    df: pd.DataFrame,
    *,
    model: Optional["transformers.pipelines.Pipeline"] = None,
    model_name: str = "openai/whisper-large-v3",
    device: Optional[Union[int, str]] = None,
    torch_dtype: Optional["torch.dtype"] = None,
    audio_column: str = "audio_path",
    transcription_column: str = "whisper_large_v3_transcription",
    segments_column: Optional[str] = "whisper_large_v3_segments",
    show_progress: bool = True,
    skip_missing: bool = False,
    transcribe_kwargs: Optional[dict] = None,
    generate_kwargs: Optional[dict] = None,
    chunk_length_s: Optional[float] = 30.0,
    stride_length_s: Optional[Union[float, tuple[float, float]]] = 5.0,
    return_timestamps: Union[bool, str] = True,
    ignore_warning: bool = True,
) -> pd.DataFrame:
    """Run Whisper inference over a DataFrame of audio paths using HF pipeline.

    The implementation mirrors the long-form transcription recipe from the
    ``openai/whisper-large-v3`` model card on Hugging Face: audio is processed in
    sliding chunks (default 30s with 5s stride), timestamp decoding is enabled,
    and the optional ``segments_column`` captures the per-chunk metadata that the
    pipeline returns.

    Notes
    -----
    This function does not mutate the input DataFrame; a copy is returned.
    """
    transcribe_kwargs = _sanitize_generate_kwargs(transcribe_kwargs)
    effective_generate_kwargs = {
        "task": "transcribe",
        "language": "de",
        **_sanitize_generate_kwargs(generate_kwargs),
    }

    if model is None:
        whisper_pipe = load_whisper_model(
            model_name=model_name,
            device=device,
            torch_dtype=torch_dtype,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            return_timestamps=return_timestamps,
            ignore_warning=ignore_warning,
            generate_kwargs=effective_generate_kwargs,
        )
    else:
        whisper_pipe = model
        existing_kwargs = _sanitize_generate_kwargs(getattr(whisper_pipe, "generate_kwargs", {}))
        whisper_pipe.generate_kwargs = {**existing_kwargs, **effective_generate_kwargs}
        if chunk_length_s is not None:
            setattr(whisper_pipe, "chunk_length_s", chunk_length_s)
        if stride_length_s is not None:
            setattr(whisper_pipe, "stride_length_s", stride_length_s)
        if return_timestamps is not None:
            setattr(whisper_pipe, "return_timestamps", return_timestamps)
        if ignore_warning is not None:
            setattr(whisper_pipe, "ignore_warning", ignore_warning)

    df_with_transcriptions = df.copy()
    transcripts: list[Optional[str]] = []
    segments_store: list[Optional[list]] | None = [] if segments_column else None

    for audio_path in _iter_audio_paths(df_with_transcriptions[audio_column], show_progress):
        if not audio_path.is_file():
            message = f"Audio file not found: {audio_path}"
            if skip_missing:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                transcripts.append(None)
                if segments_store is not None:
                    segments_store.append(None)
                continue
            raise FileNotFoundError(message)

        if audio_path.suffix.lower() == ".nis":
            message = (
                f"Skipping NIST file '{audio_path.name}'. Convert it to WAV before running Whisper."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            transcripts.append(None)
            if segments_store is not None:
                segments_store.append(None)
            continue

        call_kwargs = {**transcribe_kwargs}
        if chunk_length_s is not None:
            call_kwargs["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            call_kwargs["stride_length_s"] = stride_length_s
        if return_timestamps is not None:
            call_kwargs["return_timestamps"] = return_timestamps
        if ignore_warning is not None:
            call_kwargs["ignore_warning"] = ignore_warning

        if "generate_kwargs" in call_kwargs:
            call_kwargs["generate_kwargs"] = _sanitize_generate_kwargs(call_kwargs["generate_kwargs"])

        result = whisper_pipe(str(audio_path), **call_kwargs)
        transcripts.append(result.get("text", "").strip())
        if segments_store is not None:
            segments_store.append(result.get("segments") or result.get("chunks"))

    df_with_transcriptions[transcription_column] = transcripts
    if segments_store is not None:
        df_with_transcriptions[segments_column] = segments_store
    return df_with_transcriptions


def add_wer_column(
    df: pd.DataFrame,
    *,
    reference_column: str = "gt_transcription",
    hypothesis_column: str = "whisper_large_v3_transcription",
    output_column: str = "whisper_large_v3_wer",
    normalizer: Optional["jiwer.Compose"] = None,
) -> pd.DataFrame:
    """Attach a WER column using ``jiwer``."""
    _ensure_package("jiwer", "Install it via `pip install jiwer`.")  # pragma: no cover

    from jiwer import Compose, RemoveMultipleSpaces, ReplaceRegex, Strip, ToLowerCase, wer

    transform = normalizer or Compose(
        [
            ReplaceRegex(r"-", " "),
            ToLowerCase(),
            RemoveMultipleSpaces(),
            Strip(),
        ]
    )

    df_with_wer = df.copy()
    scores: list[Optional[float]] = []

    for truth, hypo in zip(df_with_wer[reference_column], df_with_wer[hypothesis_column]):
        if truth is None or hypo is None:
            scores.append(None)
            continue
        scores.append(
            wer(
                str(truth),
                str(hypo),
                truth_transform=transform,
                hypothesis_transform=transform,
            )
        )

    df_with_wer[output_column] = scores
    return df_with_wer


def run_whisper_large_v3_pipeline(
    df: pd.DataFrame,
    *,
    model: Optional["transformers.pipelines.Pipeline"] = None,
    device: Optional[Union[int, str]] = None,
    torch_dtype: Optional["torch.dtype"] = None,
    audio_column: str = "audio_path",
    reference_column: str = "gt_transcription",
    transcription_column: str = "whisper_large_v3_transcription",
    segments_column: Optional[str] = "whisper_large_v3_segments",
    wer_column: str = "whisper_large_v3_wer",
    show_progress: bool = True,
    skip_missing: bool = False,
    transcribe_kwargs: Optional[dict] = None,
    generate_kwargs: Optional[dict] = None,
    chunk_length_s: Optional[float] = 30.0,
    stride_length_s: Optional[Union[float, tuple[float, float]]] = 5.0,
    return_timestamps: Union[bool, str] = True,
    ignore_warning: bool = True,
    normalizer: Optional["jiwer.Compose"] = None,
) -> pd.DataFrame:
    """High-level helper: Whisper inference + WER in one call.

    Returns the input DataFrame enriched with the Whisper transcription,
    optional timestamp ``segments_column`` and a WER column computed with jiwer.
    """
    df_with_predictions = transcribe_dataframe(
        df,
        model=model,
        model_name="openai/whisper-large-v3",
        device=device,
        torch_dtype=torch_dtype,
        audio_column=audio_column,
        transcription_column=transcription_column,
        segments_column=segments_column,
        show_progress=show_progress,
        skip_missing=skip_missing,
        transcribe_kwargs=transcribe_kwargs,
        generate_kwargs=generate_kwargs,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=return_timestamps,
        ignore_warning=ignore_warning,
    )
    return add_wer_column(
        df_with_predictions,
        reference_column=reference_column,
        hypothesis_column=transcription_column,
        output_column=wer_column,
        normalizer=normalizer,
    )


__all__ = [
    "load_whisper_model",
    "transcribe_dataframe",
    "add_wer_column",
    "run_whisper_large_v3_pipeline",
]
