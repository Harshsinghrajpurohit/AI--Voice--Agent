"""Text-to-speech adapter running Piper TTS as a local subprocess."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ..config import Settings
from ..errors import SpeechSynthesisError

logger = logging.getLogger(__name__)


class PiperTTS:
    """Synthesizes clean spoken text into a WAV file using the local Piper binary."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        """Run piper subprocess to synthesize text into a wav file."""
        clean_text = text.strip()
        if not clean_text:
            raise SpeechSynthesisError("Cannot synthesize empty text.")

        out = output_path or self.settings.paths.audio_out
        piper_exe = shutil.which(self.settings.tts.piper_exe) or self.settings.tts.piper_exe
        model_path = self.settings.paths.voice_model

        if not model_path.is_file():
            raise SpeechSynthesisError(
                f"Piper voice model file not found: {model_path}. "
                "Download en_US-lessac-medium.onnx (+ .json) to the project directory."
            )

        cmd: list[str] = [
            str(piper_exe),
            "--model",
            str(model_path),
            "--output_file",
            str(out),
            "--length_scale",
            str(self.settings.tts.length_scale),
        ]

        logger.debug("Synthesizing speech via Piper: %s", cmd)
        try:
            proc = subprocess.run(
                cmd,
                input=clean_text,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                raise SpeechSynthesisError(
                    f"Piper synthesis failed (code {proc.returncode}): {proc.stderr.strip()}"
                )
        except FileNotFoundError as exc:
            raise SpeechSynthesisError(
                f"Piper executable '{self.settings.tts.piper_exe}' not found on PATH."
            ) from exc
        except Exception as exc:
            if isinstance(exc, SpeechSynthesisError):
                raise
            raise SpeechSynthesisError(f"Unexpected error running Piper TTS: {exc}") from exc

        if not out.is_file() or out.stat().st_size == 0:
            raise SpeechSynthesisError(f"Piper produced missing or empty audio output at {out}")

        return out
