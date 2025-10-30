from __future__ import annotations

import math
import warnings
from pathlib import Path
import tempfile
from typing import Optional, Union

import pandas as pd


def _ensure_package(name: str, install_hint: str) -> None:
    try:
        __import__(name)
    except ImportError as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            f"The '{name}' package is required for this operation. {install_hint}"
        ) from exc


def load_parakeet_model(
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    *,
    device: Optional[Union[int, str]] = None,
    enable_long_form: bool = True,
    att_context_size: Optional[list[int]] = None,
) -> "nemo.collections.asr.models.ASRModel":
    """Load the NeMo Parakeet ASR model for transcription.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier for Parakeet, e.g. ``nvidia/parakeet-tdt-0.6b-v3``.
    device:
        Device to load the model on. ``"cuda"`` or ``"cpu"``.
        When omitted, CUDA is used if available.
    enable_long_form:
        Whether to configure the model for long-form audio (>24 minutes).
        Switches to local attention with larger context.
    att_context_size:
        Attention context size for long-form audio. Defaults to ``[256, 256]``.

    Returns
    -------
    nemo.collections.asr.models.ASRModel
        The loaded Parakeet model ready for transcription.
    """
    _ensure_package("nemo", "Install it via `pip install nemo_toolkit[asr]`.")
    _ensure_package("torch", "Install PyTorch compatible with your platform.")
    _ensure_package("datasets", "Install it via `pip install datasets`.")

    import nemo.collections.asr as nemo_asr
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Move model to specified device
    if device == "cuda" or (isinstance(device, int) and device >= 0):
        asr_model = asr_model.cuda()
    else:
        asr_model = asr_model.cpu()

    # Configure for long-form audio if requested
    if enable_long_form:
        context = att_context_size or [256, 256]
        try:
            asr_model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=context,
            )
        except (AttributeError, TypeError):
            # Model might not support this configuration
            warnings.warn(
                "Could not configure long-form attention. Model may have issues with audio >24 minutes.",
                RuntimeWarning,
                stacklevel=2,
            )

    return asr_model


try:  # pragma: no cover - optional dependency
    from audio_utils import InMemoryAudio as _InMemoryAudio
except ImportError:  # pragma: no cover - fallback when module unavailable
    _InMemoryAudio = None


def transcribe_dataframe(
    df: pd.DataFrame,
    *,
    model: Optional["nemo.collections.asr.models.ASRModel"] = None,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: Optional[Union[int, str]] = None,
    audio_column: str = "audio_path",
    transcription_column: str = "parakeet_tdt_v3_transcription",
    segments_column: Optional[str] = "parakeet_tdt_v3_segments",
    batch_size: int = 16,
    show_progress: bool = True,
    skip_missing: bool = False,
    enable_timestamps: bool = True,
    enable_long_form: bool = True,
) -> pd.DataFrame:
    """Run Parakeet ASR inference over a DataFrame of audio paths.

    Parameters
    ----------
    df:
        Input DataFrame containing audio paths.
    model:
        Pre-loaded Parakeet model. If None, a new model is loaded.
    model_name:
        Model identifier to load if model is None.
    device:
        Device for inference ("cuda" or "cpu").
    audio_column:
        Name of the column containing audio file paths.
    transcription_column:
        Name of the output column for transcriptions.
    segments_column:
        Name of the output column for timestamp segments. Set to None to disable.
    batch_size:
        Number of audio files to process in a single batch. Larger values improve
        throughput but require more GPU memory. Default is 16.
    show_progress:
        Whether to display a progress bar.
    skip_missing:
        Whether to skip missing audio files (with warning) or raise an error.
    enable_timestamps:
        Whether to request timestamp information from the model.
    enable_long_form:
        Whether to configure the model for long-form audio.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added transcription column(s).

    Notes
    -----
    This function does not mutate the input DataFrame; a copy is returned.
    """
    if model is None:
        parakeet_model = load_parakeet_model(
            model_name=model_name,
            device=device,
            enable_long_form=enable_long_form,
        )
    else:
        parakeet_model = model

    df_with_transcriptions = df.copy()
    audio_entries = list(df_with_transcriptions[audio_column])
    row_count = len(audio_entries)

    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        try:
            if math.isnan(value):  # type: ignore[arg-type]
                return True
        except TypeError:
            pass
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed == "" or trimmed in {"[]", "{}"}
        if isinstance(value, (list, tuple, set)):
            return len(value) == 0
        return False

    if transcription_column in df_with_transcriptions.columns:
        existing_transcripts = df_with_transcriptions[transcription_column].tolist()
        transcripts = [None if _is_missing(value) else value for value in existing_transcripts]
    else:
        transcripts = [None] * row_count

    if segments_column:
        if segments_column in df_with_transcriptions.columns:
            existing_segments = df_with_transcriptions[segments_column].tolist()
            segments_store = [None if _is_missing(value) else value for value in existing_segments]
        else:
            segments_store = [None] * row_count
    else:
        segments_store = None

    valid_indices: list[int] = []
    valid_paths: list[str] = []
    temp_files: list[Path] = []

    for idx, entry in enumerate(audio_entries):
        existing_transcript = transcripts[idx] if idx < len(transcripts) else None
        existing_segments = segments_store[idx] if segments_store is not None else None
        needs_transcription = _is_missing(existing_transcript)
        needs_segments = segments_store is not None and _is_missing(existing_segments)

        if not (needs_transcription or needs_segments):
            continue

        if entry is None:
            continue

        if _InMemoryAudio is not None and isinstance(entry, _InMemoryAudio):
            try:
                wav_bytes = entry.to_wav_bytes()
            except Exception as exc:
                warnings.warn(
                    f"Failed to materialise in-memory audio for entry {entry.descriptor()}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                temp_file.write(wav_bytes)
                temp_file.flush()
            finally:
                temp_file.close()

            temp_path = Path(temp_file.name)
            temp_files.append(temp_path)
            valid_indices.append(idx)
            valid_paths.append(str(temp_path))
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
                f"Skipping NIST file '{audio_path.name}'. Convert it to WAV before running Parakeet."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue

        valid_indices.append(idx)
        valid_paths.append(str(audio_path))

    num_batches = (len(valid_paths) + batch_size - 1) // batch_size
    batch_iterator = range(num_batches)

    if show_progress:
        try:
            from tqdm import tqdm

            batch_iterator = tqdm(batch_iterator, desc="Transcribing (Parakeet)", unit="batch")
        except ImportError:
            pass

    try:
        for batch_idx in batch_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(valid_paths))
            batch_paths = valid_paths[start_idx:end_idx]

            try:
                result = parakeet_model.transcribe(
                    batch_paths,
                    batch_size=batch_size,
                    timestamps=enable_timestamps,
                )
            except Exception as exc:
                warnings.warn(
                    f"Batch transcription failed for batch {batch_idx}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            if isinstance(result, list):
                for offset, transcription_obj in enumerate(result):
                    target_idx = valid_indices[start_idx + offset]
                    text = getattr(transcription_obj, "text", "")
                    transcripts[target_idx] = text.strip() if text else None

                    if segments_store is not None and enable_timestamps:
                        timestamp_info = getattr(transcription_obj, "timestamp", None)
                        if timestamp_info:
                            segments = timestamp_info.get("segment", [])
                            segments_store[target_idx] = segments if segments else None
                        else:
                            segments_store[target_idx] = None
            else:
                for offset in range(len(batch_paths)):
                    target_idx = valid_indices[start_idx + offset]
                    transcripts[target_idx] = None
                    if segments_store is not None:
                        segments_store[target_idx] = None
    finally:
        for temp_path in temp_files:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                warnings.warn(f"Failed to remove temporary file {temp_path}", RuntimeWarning, stacklevel=2)

    df_with_transcriptions[transcription_column] = transcripts
    if segments_store is not None:
        df_with_transcriptions[segments_column] = segments_store
    return df_with_transcriptions


def run_parakeet_pipeline(
    df: pd.DataFrame,
    *,
    model: Optional["nemo.collections.asr.models.ASRModel"] = None,
    device: Optional[Union[int, str]] = None,
    audio_column: str = "audio_path",
    reference_column: str = "gt_transcription",
    transcription_column: str = "parakeet_tdt_v3_transcription",
    segments_column: Optional[str] = "parakeet_tdt_v3_segments",
    wer_column: str = "parakeet_tdt_v3_wer",
    batch_size: int = 16,
    show_progress: bool = True,
    skip_missing: bool = False,
    enable_timestamps: bool = True,
    enable_long_form: bool = True,
    normalizer: Optional["jiwer.Compose"] = None,
) -> pd.DataFrame:
    """High-level helper: Parakeet inference + WER in one call.

    Returns the input DataFrame enriched with Parakeet transcription,
    optional timestamp segments, and a WER column computed with jiwer.

    Parameters
    ----------
    df:
        Input DataFrame with audio paths and ground truth transcriptions.
    model:
        Pre-loaded Parakeet model (optional).
    device:
        Device for inference.
    audio_column:
        Column name containing audio file paths.
    reference_column:
        Column name containing ground truth transcriptions.
    transcription_column:
        Output column name for Parakeet transcriptions.
    segments_column:
        Output column name for timestamp segments.
    wer_column:
        Output column name for WER scores.
    batch_size:
        Number of audio files to process in a single batch.
    show_progress:
        Whether to show progress bar.
    skip_missing:
        Whether to skip missing files.
    enable_timestamps:
        Whether to extract timestamps.
    enable_long_form:
        Whether to enable long-form audio support.
    normalizer:
        Optional jiwer normalizer for WER calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with transcriptions and WER scores.
    """
    # Import add_wer_column from whisper_pipeline
    from whisper_pipeline import add_wer_column

    df_with_predictions = transcribe_dataframe(
        df,
        model=model,
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        device=device,
        audio_column=audio_column,
        transcription_column=transcription_column,
        segments_column=segments_column,
        batch_size=batch_size,
        show_progress=show_progress,
        skip_missing=skip_missing,
        enable_timestamps=enable_timestamps,
        enable_long_form=enable_long_form,
    )

    return add_wer_column(
        df_with_predictions,
        reference_column=reference_column,
        hypothesis_column=transcription_column,
        output_column=wer_column,
        normalizer=normalizer,
    )


__all__ = [
    "load_parakeet_model",
    "transcribe_dataframe",
    "run_parakeet_pipeline",
]
