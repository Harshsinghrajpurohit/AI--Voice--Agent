"""Speech-to-text adapter wrapping faster-whisper with domain vocabulary biasing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, Sequence
import numpy as np

from ..config import Settings
from ..errors import TranscriptionError

logger = logging.getLogger(__name__)


class WhisperModelProtocol(Protocol):
    """Protocol for faster-whisper WhisperModel to enable mocking in tests."""

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        **kwargs: Any,
    ) -> tuple[Sequence[Any], Any]: ...


class Transcriber:
    """Loads and runs Faster-Whisper with banking vocabulary biasing."""

    def __init__(
        self,
        settings: Settings,
        model: WhisperModelProtocol | None = None,
    ) -> None:
        self.settings = settings
        self._model = model

    def _get_model(self) -> WhisperModelProtocol:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading Faster-Whisper model size=%s device=%s compute_type=%s",
                    self.settings.stt.model_size,
                    self.settings.stt.device,
                    self.settings.stt.compute_type,
                )
                self._model = WhisperModel(
                    self.settings.stt.model_size,
                    device=self.settings.stt.device,
                    compute_type=self.settings.stt.compute_type,
                )
            except Exception as exc:
                raise TranscriptionError(f"Failed to load Faster-Whisper model: {exc}") from exc
        return self._model

    def transcribe(self, audio: str | Path | np.ndarray) -> str:
        """Transcribe audio file or waveform array to text using domain initial prompt."""
        if isinstance(audio, Path):
            audio = str(audio)

        try:
            model = self._get_model()
            segments, _info = model.transcribe(
                audio,
                language=self.settings.stt.language,
                initial_prompt=self.settings.stt.initial_prompt,
                beam_size=5,
            )
            raw_text = " ".join(seg.text for seg in segments)
            text = " ".join(raw_text.split())
            logger.debug("Transcribed text: %s", text)
            return text
        except Exception as exc:
            if isinstance(exc, TranscriptionError):
                raise
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
