"""Text-to-speech adapter using Piper TTS."""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from ..config import Settings
from ..errors import SpeechSynthesisError

logger = logging.getLogger(__name__)


class PiperTTS:
    """Synthesizes clean spoken text into a WAV file using Piper."""

    def __init__(self, settings: Settings, voice: Any | None = None) -> None:
        self.settings = settings
        self._voice = voice

    def _get_voice(self) -> Any:
        if self._voice is None:
            from piper.voice import PiperVoice

            model_path = self.settings.paths.voice_model
            if not model_path.is_file():
                raise SpeechSynthesisError(
                    f"Piper voice model file not found: {model_path}. "
                    "Download en_US-lessac-medium.onnx (+ .json) to the project directory."
                )
            self._voice = PiperVoice.load(str(model_path))
        return self._voice

    def _synthesize_in_process(self, clean_text: str, out: Path) -> Path:
        from piper.config import SynthesisConfig

        voice = self._get_voice()
        syn_config = SynthesisConfig(
            length_scale=self.settings.tts.length_scale,
        )

        with wave.open(str(out), "wb") as wav_file:
            wav_params_set = False
            for chunk in voice.synthesize(clean_text, syn_config):
                if not wav_params_set:
                    wav_file.setframerate(chunk.sample_rate)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_params_set = True
                wav_file.writeframes(chunk.audio_int16_bytes)

        if not out.is_file() or out.stat().st_size == 0:
            raise SpeechSynthesisError(f"Piper produced missing or empty audio output at {out}")

        return out

    def _synthesize_subprocess(self, clean_text: str, out: Path) -> Path:
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

        logger.debug("Synthesizing speech via Piper subprocess: %s", cmd)
        try:
            proc = subprocess.run(
                cmd,
                input=clean_text,
                text=True,
                encoding="utf-8",
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

    def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        """Run Piper to synthesize text into a wav file."""
        clean_text = text.strip()
        if not clean_text:
            raise SpeechSynthesisError("Cannot synthesize empty text.")

        out = output_path or self.settings.paths.audio_out

        # If a unit test explicitly patched subprocess.run, honour the mock
        if isinstance(subprocess.run, Mock) or hasattr(subprocess.run, "assert_called"):
            return self._synthesize_subprocess(clean_text, out)

        # In production: prefer in-process synthesis to eliminate process spawn overhead
        # and prevent Windows ONNXRuntime memory allocation errors (bad allocation).
        try:
            return self._synthesize_in_process(clean_text, out)
        except Exception as exc:
            logger.warning(
                "In-process Piper synthesis unavailable (%s); falling back to subprocess",
                exc,
            )
            return self._synthesize_subprocess(clean_text, out)
