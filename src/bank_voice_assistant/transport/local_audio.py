"""Local audio capture and playback using sounddevice and WebRTC VAD."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read, write

from ..config import Settings, VadSettings
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


def frame_rms(frame: np.ndarray) -> float:
    """Root-mean-square level of one int16 frame, in raw sample counts."""
    data = frame.astype(np.float64)
    return float(np.sqrt(np.mean(data * data)))


def ambient_rms(frame_levels: Sequence[float]) -> float:
    """Robust ambient level for a calibration window: the 20th percentile.

    A spoken word inside the window raises the mean but not the low end, so the
    percentile is what stops a talker calibrating the gate to their own voice.
    """
    if not frame_levels:
        return 0.0
    return float(np.percentile(np.asarray(frame_levels, dtype=np.float64), 20))


def energy_gate(ambient: float, vad: VadSettings) -> float:
    """RMS a frame must clear to count as speech, for this microphone.

    A fixed gate only ever suits one microphone: speech from a quiet internal
    array sits just above its noise floor, while a boom headset is far louder.
    Tracking the ambient noise (``noise_margin``) between an absolute floor and
    the configured ``energy_threshold`` makes one setting serve both, with
    headphones or not. ``calibration_ms = 0`` restores the fixed gate.
    """
    if vad.calibration_ms <= 0:
        return float(vad.energy_threshold)
    adaptive = ambient * vad.noise_margin
    return float(min(vad.energy_threshold, max(vad.energy_floor, adaptive)))


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
        calibration_frames = max(0, cfg_vad.calibration_ms // cfg_audio.frame_ms)

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
                gate = self._calibrate_gate(stream, frame_samples, calibration_frames, cfg_vad)
                for _ in range(max_frames):
                    frame, overflowed = stream.read(frame_samples)
                    if overflowed:
                        logger.warning("Microphone buffer overflowed.")

                    frame_bytes = frame.tobytes()
                    frame_energy = frame_rms(frame)
                    is_speech = (
                        vad.is_speech(frame_bytes, cfg_audio.sample_rate)
                        and frame_energy > gate
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

    def _calibrate_gate(
        self,
        stream: object,
        frame_samples: int,
        calibration_frames: int,
        cfg_vad: VadSettings,
    ) -> float:
        """Choose this turn's RMS gate from the ambient noise of the live microphone.

        Measured before listening on every turn, so plugging in or unplugging a
        headset simply changes the device the OS reports and the gate adapts next
        turn: one setting serves a quiet internal array and a loud boom mic alike.
        """
        if calibration_frames <= 0:
            return float(cfg_vad.energy_threshold)
        levels = [frame_rms(stream.read(frame_samples)[0]) for _ in range(calibration_frames)]
        ambient = ambient_rms(levels)
        gate = energy_gate(ambient, cfg_vad)
        logger.info("Ambient noise %.0f RMS -> speaking gate %.0f RMS", ambient, gate)
        return gate

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

    def describe_devices(self) -> dict[str, str]:
        """Name the default input/output devices. ``--doctor`` uses this check.

        Raises:
            AudioError: when PortAudio reports no usable default device.
        """
        import sounddevice as sd

        try:
            input_device = sd.query_devices(kind="input")
            output_device = sd.query_devices(kind="output")
        except Exception as exc:
            raise AudioError(f"No usable default audio device: {exc}") from exc

        return {
            "input": f"{input_device['name']} ({int(input_device['max_input_channels'])} ch)",
            "output": f"{output_device['name']} ({int(output_device['max_output_channels'])} ch)",
        }
