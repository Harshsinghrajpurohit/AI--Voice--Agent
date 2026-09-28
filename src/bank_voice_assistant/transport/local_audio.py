"""Local audio capture and playback using sounddevice and WebRTC VAD."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read, write

from ..config import Settings
from ..errors import AudioError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioCaptureResult:
    """Result of a microphone recording session."""

    audio: np.ndarray
    sample_rate: int
    duration_s: float
    max_amplitude: int
    speech_detected: bool


class AudioTransport:
    """Manages audio I/O: microphone recording with VAD endpointing and speaker playback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def record_turn(self, output_file: Path | None = None) -> AudioCaptureResult:
        """Capture microphone audio with WebRTC VAD endpointing and RMS energy gating."""
        import sounddevice as sd
        import webrtcvad

        cfg_audio = self.settings.audio
        cfg_vad = self.settings.vad

        vad = webrtcvad.Vad(cfg_vad.aggressiveness)
        frame_samples = cfg_audio.frame_samples
        max_frames = int(cfg_vad.max_record_seconds * 1000 / cfg_audio.frame_ms)
        silence_frames_to_stop = cfg_vad.silence_ms_to_stop // cfg_audio.frame_ms

        frames: list[bytes] = []
        speech_started = False
        consecutive_speech_frames = 0
        consecutive_silence_frames = 0

        logger.info("Listening for speech (sample_rate=%d)...", cfg_audio.sample_rate)

        try:
            with sd.InputStream(
                samplerate=cfg_audio.sample_rate,
                channels=cfg_audio.channels,
                dtype=cfg_audio.dtype,
                blocksize=frame_samples,
            ) as stream:
                for _ in range(max_frames):
                    frame, overflowed = stream.read(frame_samples)
                    if overflowed:
                        logger.warning("Microphone buffer overflowed.")

                    frame_bytes = frame.tobytes()
                    frame_energy = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                    is_speech = (
                        vad.is_speech(frame_bytes, cfg_audio.sample_rate)
                        and frame_energy > cfg_vad.energy_threshold
                    )

                    if not speech_started:
                        if is_speech:
                            consecutive_speech_frames += 1
                            frames.append(frame_bytes)
                            if consecutive_speech_frames >= cfg_vad.min_speech_frames:
                                speech_started = True
                        else:
                            consecutive_speech_frames = 0
                            frames.clear()
                    else:
                        frames.append(frame_bytes)
                        if is_speech:
                            consecutive_silence_frames = 0
                        else:
                            consecutive_silence_frames += 1
                            if consecutive_silence_frames >= silence_frames_to_stop:
                                break
        except Exception as exc:
            raise AudioError(f"Microphone capture failed: {exc}") from exc

        if not frames:
            empty_audio = np.array([], dtype=np.int16)
            return AudioCaptureResult(
                audio=empty_audio,
                sample_rate=cfg_audio.sample_rate,
                duration_s=0.0,
                max_amplitude=0,
                speech_detected=False,
            )

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        duration_s = len(audio) / cfg_audio.sample_rate
        max_amp = int(np.abs(audio).max()) if len(audio) > 0 else 0

        target_file = output_file or self.settings.paths.audio_in
        try:
            write(str(target_file), cfg_audio.sample_rate, audio)
        except Exception as exc:
            raise AudioError(f"Failed to write recorded audio to {target_file}: {exc}") from exc

        return AudioCaptureResult(
            audio=audio,
            sample_rate=cfg_audio.sample_rate,
            duration_s=duration_s,
            max_amplitude=max_amp,
            speech_detected=speech_started,
        )

    def play_audio(self, audio_source: Path | np.ndarray, sample_rate: int | None = None) -> None:
        """Play audio file or numpy array through the system speakers."""
        import sounddevice as sd

        try:
            if isinstance(audio_source, (str, Path)):
                sr, audio_data = read(str(audio_source))
            else:
                sr = sample_rate or self.settings.audio.sample_rate
                audio_data = audio_source

            sd.play(audio_data, sr)
            sd.wait()
        except Exception as exc:
            raise AudioError(f"Audio playback failed: {exc}") from exc
