from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Union

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


def _build_german_number_map() -> dict[str, str]:
    """Build a mapping of German number words (0-100) to digits.

    Supports multiple spelling variants:
    - ß/ss: dreißig/dreissig
    - ü/ue: fünf/fuenf
    - ö/oe: zwölf/zwoelf
    """
    number_map = {
        # 0-20 (standard spellings)
        "null": "0",
        "eins": "1",
        "zwei": "2",
        "drei": "3",
        "vier": "4",
        "fünf": "5",
        "sechs": "6",
        "sieben": "7",
        "acht": "8",
        "neun": "9",
        "zehn": "10",
        "elf": "11",
        "zwölf": "12",
        "dreizehn": "13",
        "vierzehn": "14",
        "fünfzehn": "15",
        "sechzehn": "16",
        "siebzehn": "17",
        "achtzehn": "18",
        "neunzehn": "19",
        "zwanzig": "20",
        # Tens (standard spellings)
        "dreißig": "30",
        "vierzig": "40",
        "fünfzig": "50",
        "sechzig": "60",
        "siebzig": "70",
        "achtzig": "80",
        "neunzig": "90",
        "hundert": "100",
        # Alternative spellings (ue/oe/ss variants)
        "fuenf": "5",
        "zwoelf": "12",
        "fuenfzehn": "15",
        "dreissig": "30",
        "fuenfzig": "50",
    }

    # Generate compound numbers with ALL spelling variants
    # Ones: standard and alternative spellings
    ones_standard = ["ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
    ones_alternatives = {
        "fünf": ["fuenf"],  # ü variant
    }

    # Tens: standard and alternative spellings
    tens_standard = ["zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
    tens_alternatives = {
        "dreißig": ["dreissig"],  # ß variant
        "fünfzig": ["fuenfzig"],  # ü variant
    }
    tens_values = [20, 30, 40, 50, 60, 70, 80, 90]

    # Generate all combinations: {one}und{ten} with all variants
    for one_word, one_val in zip(ones_standard, range(1, 10)):
        # Get all variants for this "one" word (standard + alternatives)
        one_variants = [one_word] + ones_alternatives.get(one_word, [])

        for ten_word, ten_val in zip(tens_standard, tens_values):
            # Get all variants for this "ten" word (standard + alternatives)
            ten_variants = [ten_word] + tens_alternatives.get(ten_word, [])

            # Create compound for all combinations
            for one_variant in one_variants:
                for ten_variant in ten_variants:
                    compound = f"{one_variant}und{ten_variant}"
                    number_map[compound] = str(ten_val + one_val)

    return number_map


def _normalize_german_numbers(text: str) -> str:
    """Replace German number words (0-100) with digits."""
    number_map = _build_german_number_map()

    # Sort by length (longest first) to match compound numbers before components
    sorted_numbers = sorted(number_map.keys(), key=len, reverse=True)

    # Build regex pattern with word boundaries
    pattern = r"\b(" + "|".join(re.escape(num) for num in sorted_numbers) + r")\b"

    def replace_number(match: re.Match[str]) -> str:
        word = match.group(1)
        return number_map.get(word, word)

    return re.sub(pattern, replace_number, text, flags=re.IGNORECASE)


def _remove_punctuation(text: str) -> str:
    """Remove punctuation marks from text."""
    import string
    # Remove all punctuation
    return text.translate(str.maketrans("", "", string.punctuation))


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

    # Clear deprecated forced decoder ids so that language/task flags control generation fully.
    generation_config = getattr(pipe.model, "generation_config", None)
    if generation_config is not None:
        generation_config.forced_decoder_ids = None
    if getattr(pipe.model, "config", None) is not None:
        pipe.model.config.forced_decoder_ids = None

    if generate_kwargs:
        existing_kwargs = _sanitize_generate_kwargs(getattr(pipe, "generate_kwargs", {}))
        pipe.generate_kwargs = {**existing_kwargs, **_sanitize_generate_kwargs(generate_kwargs)}

    return pipe


try:  # pragma: no cover - optional dependency
    from audio_utils import InMemoryAudio as _InMemoryAudio
except ImportError:  # pragma: no cover - fallback when module unavailable
    _InMemoryAudio = None


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
    batch_size: Optional[int] = 16,
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
    ``batch_size`` controls how many audio items are forwarded to the pipeline
    per inference call. Increasing it improves GPU utilisation at the cost of
    additional memory usage.

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
    audio_values = list(df_with_transcriptions[audio_column])
    transcripts: list[Optional[str]] = [None] * len(audio_values)
    segments_store: list[Optional[list]] | None = [None] * len(audio_values) if segments_column else None

    call_kwargs_base: dict[str, object] = {**transcribe_kwargs}
    if chunk_length_s is not None:
        call_kwargs_base["chunk_length_s"] = chunk_length_s
    if stride_length_s is not None:
        call_kwargs_base["stride_length_s"] = stride_length_s
    if return_timestamps is not None:
        call_kwargs_base["return_timestamps"] = return_timestamps
    if ignore_warning is not None:
        call_kwargs_base["ignore_warning"] = ignore_warning
    if "generate_kwargs" in call_kwargs_base:
        call_kwargs_base["generate_kwargs"] = _sanitize_generate_kwargs(call_kwargs_base["generate_kwargs"])

    prepared_inputs: list[Union[str, dict[str, object]]] = []
    valid_indices: list[int] = []

    for idx, entry in enumerate(audio_values):
        if entry is None:
            continue

        if _InMemoryAudio is not None and isinstance(entry, _InMemoryAudio):
            prepared_inputs.append(entry.to_whisper_input())
            valid_indices.append(idx)
            continue

        audio_path = Path(str(entry))
        if not audio_path.is_file():
            message = f"Audio file not found: {audio_path}"
            if skip_missing:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue
            raise FileNotFoundError(message)

        if audio_path.suffix.lower() == ".nis":
            message = (
                f"Skipping NIST file '{audio_path.name}'. Convert it to WAV before running Whisper."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue

        prepared_inputs.append(str(audio_path))
        valid_indices.append(idx)

    if not prepared_inputs:
        return df_with_transcriptions

    effective_batch_size = max(1, batch_size or 1)

    progress_bar = None
    if show_progress:
        try:  # pragma: no cover - optional dependency
            from tqdm import tqdm

            progress_bar = tqdm(total=len(prepared_inputs), desc="Transcribing", unit="file")
        except ImportError:
            progress_bar = None

    try:
        for start in range(0, len(prepared_inputs), effective_batch_size):
            end = min(start + effective_batch_size, len(prepared_inputs))
            batch_inputs = prepared_inputs[start:end]
            batch_indices = valid_indices[start:end]

            call_kwargs = dict(call_kwargs_base)
            call_kwargs.setdefault("batch_size", effective_batch_size)

            result = whisper_pipe(batch_inputs, **call_kwargs)
            if isinstance(result, dict):
                result_items = [result]
            else:
                result_items = list(result)

            for item, target_idx in zip(result_items, batch_indices):
                text = item.get("text", "").strip()
                transcripts[target_idx] = text or None
                if segments_store is not None:
                    segments_store[target_idx] = item.get("segments") or item.get("chunks")

            if progress_bar is not None:
                progress_bar.update(len(batch_indices))
    finally:
        if progress_bar is not None:
            progress_bar.close()

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
    """Attach a WER column using ``jiwer``.

    Text normalization includes:
    - Lowercase conversion
    - German number words (0-100) to digits (e.g., "drei" → "3", "einundzwanzig" → "21")
    - Punctuation removal
    - Hyphen replacement with spaces
    - Multiple space removal
    """
    _ensure_package("jiwer", "Install it via `pip install jiwer`.")  # pragma: no cover

    from jiwer import wer

    def _preprocess_text(text: str) -> str:
        """Apply custom preprocessing: lowercase, normalize numbers, remove punctuation."""
        value = str(text).lower()
        value = _normalize_german_numbers(value)
        value = _remove_punctuation(value)
        return value

    transform = normalizer
    fallback_regex: Optional[re.Pattern[str]] = None
    use_transform_args = True

    if transform is None:
        try:
            from jiwer import Compose, RemoveMultipleSpaces, ReplaceRegex, Strip, ToLowerCase

            # Note: ToLowerCase, number normalization, and punctuation removal
            # are handled in _preprocess_text, so we only need remaining transforms
            transform = Compose(
                [
                    ReplaceRegex(r"-", " "),
                    RemoveMultipleSpaces(),
                    Strip(),
                ]
            )
        except (ImportError, AttributeError):
            try:
                from jiwer import Compose, RemoveMultipleSpaces, Strip

                transform = Compose(
                    [
                        RemoveMultipleSpaces(),
                        Strip(),
                    ]
                )
                fallback_regex = re.compile(r"-")
                use_transform_args = False
            except (ImportError, AttributeError):
                fallback_regex = re.compile(r"-")
                transform = None
                use_transform_args = False

    def _normalize_text(text: str) -> str:
        """Full normalization for fallback path (when not using jiwer transforms)."""
        value = _preprocess_text(text)
        if fallback_regex is not None:
            value = fallback_regex.sub(" ", value)
        if transform is not None and not use_transform_args:
            value = transform(value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    df_with_wer = df.copy()
    scores: list[Optional[float]] = []

    for truth, hypo in zip(df_with_wer[reference_column], df_with_wer[hypothesis_column]):
        if truth is None or hypo is None:
            scores.append(None)
            continue

        # Preprocess both texts before applying jiwer transforms
        truth_preprocessed = _preprocess_text(str(truth))
        hypo_preprocessed = _preprocess_text(str(hypo))

        if use_transform_args and transform is not None:
            scores.append(
                wer(
                    truth_preprocessed,
                    hypo_preprocessed,
                    truth_transform=transform,
                    hypothesis_transform=transform,
                )
            )
        else:
            scores.append(wer(_normalize_text(truth), _normalize_text(hypo)))

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
    batch_size: Optional[int] = 16,
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
        batch_size=batch_size,
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
