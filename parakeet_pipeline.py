from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

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


def _iter_audio_paths(paths: Iterable[str | Path], show_progress: bool) -> Iterable[Path]:
    iterator = [Path(p) for p in paths]
    if show_progress:
        try:  # pragma: no cover - optional dependency
            from tqdm import tqdm

            return (Path(p) for p in tqdm(iterator, desc="Transcribing (Parakeet)", unit="file"))
        except ImportError:
            pass
    return (Path(p) for p in iterator)


def transcribe_dataframe(
    df: pd.DataFrame,
    *,
    model: Optional["nemo.collections.asr.models.ASRModel"] = None,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: Optional[Union[int, str]] = None,
    audio_column: str = "audio_path",
    transcription_column: str = "parakeet_tdt_v3_transcription",
    segments_column: Optional[str] = "parakeet_tdt_v3_segments",
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

        # Check for NIST format (not supported by most models)
        if audio_path.suffix.lower() == ".nis":
            message = (
                f"Skipping NIST file '{audio_path.name}'. Convert it to WAV before running Parakeet."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            transcripts.append(None)
            if segments_store is not None:
                segments_store.append(None)
            continue

        # Run inference
        try:
            result = parakeet_model.transcribe(
                paths2audio_files=[str(audio_path)],
                timestamps=enable_timestamps,
            )

            # Extract transcription text
            if isinstance(result, list) and len(result) > 0:
                transcription_obj = result[0]
                # NeMo returns objects with .text attribute
                text = getattr(transcription_obj, "text", "")
                transcripts.append(text.strip() if text else None)

                # Extract timestamp information if available
                if segments_store is not None and enable_timestamps:
                    timestamp_info = getattr(transcription_obj, "timestamp", None)
                    if timestamp_info:
                        # Store segment-level timestamps
                        segments = timestamp_info.get("segment", [])
                        segments_store.append(segments if segments else None)
                    else:
                        segments_store.append(None)
            else:
                transcripts.append(None)
                if segments_store is not None:
                    segments_store.append(None)
        except Exception as e:
            warnings.warn(
                f"Transcription failed for {audio_path.name}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            transcripts.append(None)
            if segments_store is not None:
                segments_store.append(None)

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
