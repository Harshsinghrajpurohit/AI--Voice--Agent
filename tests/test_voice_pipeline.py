"""Tests for STT, TTS, and Audio Transport modules (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from bank_voice_assistant.config import Settings
from bank_voice_assistant.errors import AudioError, SpeechSynthesisError, TranscriptionError
from bank_voice_assistant.stt import Transcriber
from bank_voice_assistant.transport import AudioTransport
from bank_voice_assistant.tts import PiperTTS


# ---------------------------------------------------------------------------
# STT Tests
# ---------------------------------------------------------------------------

@dataclass
class FakeSegment:
    text: str


class FakeWhisperModel:
    def __init__(self, segments: list[str]) -> None:
        self.segments = [FakeSegment(s) for s in segments]
        self.last_kwargs: dict = {}
        self.last_audio: object = None

    def transcribe(self, audio, **kwargs):
        self.last_audio = audio
        self.last_kwargs = kwargs
        return self.segments, MagicMock()


def test_transcriber_combines_segments() -> None:
    settings = Settings()
    fake_model = FakeWhisperModel(["Hello,", " what is the home loan", " interest rate?"])
    transcriber = Transcriber(settings, model=fake_model)

    res = transcriber.transcribe("dummy.wav")
    assert res == "Hello, what is the home loan interest rate?"
    assert fake_model.last_kwargs["initial_prompt"] == settings.stt.initial_prompt
    assert fake_model.last_kwargs["language"] == "en"


def test_transcriber_scales_int16_capture_to_a_float_waveform() -> None:
    """record_turn yields int16 PCM; faster-whisper needs float32 in [-1, 1]."""
    model = FakeWhisperModel(["ok"])
    transcriber = Transcriber(Settings(), model=model)
    captured = np.ones(1600, dtype=np.int16) * 16384     # half of int16 full scale

    transcriber.transcribe(captured)

    passed = model.last_audio
    assert passed.dtype == np.float32
    assert passed.max() == pytest.approx(0.5, abs=1e-3)


def test_transcriber_hands_a_file_path_to_whisper_as_a_string() -> None:
    model = FakeWhisperModel(["ok"])
    transcriber = Transcriber(Settings(), model=model)

    transcriber.transcribe(Path("audio.wav"))

    assert model.last_audio == "audio.wav"


def test_transcriber_wraps_errors() -> None:
    settings = Settings()
    failing_model = MagicMock()
    failing_model.transcribe.side_effect = RuntimeError("Whisper crash")
    transcriber = Transcriber(settings, model=failing_model)

    with pytest.raises(TranscriptionError, match="Transcription failed"):
        transcriber.transcribe("dummy.wav")


# ---------------------------------------------------------------------------
# TTS Tests
# ---------------------------------------------------------------------------

def test_tts_empty_text_raises() -> None:
    tts = PiperTTS(Settings())
    with pytest.raises(SpeechSynthesisError, match="Cannot synthesize empty text"):
        tts.synthesize("   ")


def test_tts_missing_model_raises(tmp_path: Path) -> None:
    settings = Settings(paths=Settings().paths)
    object.__setattr__(settings.paths, "voice_model", tmp_path / "non_existent.onnx")
    tts = PiperTTS(settings)
    with pytest.raises(SpeechSynthesisError, match="voice model file not found"):
        tts.synthesize("Hello world", output_path=tmp_path / "out.wav")

def test_tts_successful_subprocess(tmp_path: Path) -> None:
    model_file = tmp_path / "model.onnx"
    model_file.write_text("dummy-onnx-content")
    out_file = tmp_path / "out.wav"

    settings = Settings()
    tts = PiperTTS(settings)

    def fake_subprocess_run(cmd, **kwargs):
        out_file.write_bytes(b"RIFF....WAVE")
        return MagicMock(returncode=0)

    with patch("bank_voice_assistant.tts.shutil.which", return_value="piper"), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        res = tts.synthesize("Hello customer", output_path=out_file)
        assert res == out_file
        assert out_file.exists()


def test_tts_sends_utf8_to_piper_so_rupee_signs_survive(tmp_path: Path) -> None:
    """Windows' cp1252 default cannot encode the rupee sign, which every rates answer has."""
    out_file = tmp_path / "out.wav"
    tts = PiperTTS(Settings())
    captured: dict = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured.update(kwargs)
        out_file.write_bytes(b"RIFF....WAVE")
        return MagicMock(returncode=0)

    with patch("bank_voice_assistant.tts.shutil.which", return_value="piper"), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        tts.synthesize("The minimum balance is \u20b95,000.", output_path=out_file)

    assert captured["encoding"] == "utf-8"
    assert "\u20b9" in captured["input"]


def test_tts_subprocess_failure_raises(tmp_path: Path) -> None:
    out_file = tmp_path / "out.wav"
    settings = Settings()
    tts = PiperTTS(settings)

    def fake_subprocess_run(cmd, **kwargs):
        return MagicMock(returncode=1, stderr="Subprocess error")

    with patch("bank_voice_assistant.tts.shutil.which", return_value="piper"), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("subprocess.run", side_effect=fake_subprocess_run):
        with pytest.raises(SpeechSynthesisError, match="Piper synthesis failed"):
            tts.synthesize("Hello", output_path=out_file)


# ---------------------------------------------------------------------------
# Audio Transport Tests
# ---------------------------------------------------------------------------

def test_audio_transport_record_speech_detected(tmp_path: Path) -> None:
    settings = Settings()
    transport = AudioTransport(settings)

    frame_samples = settings.audio.frame_samples
    speech_frame = np.ones((frame_samples, 1), dtype=np.int16) * 2000
    silence_frame = np.zeros((frame_samples, 1), dtype=np.int16)

    stream_mock = MagicMock()
    # A turn first measures ambient noise (the calibration window) before it listens,
    # so the fake stream leads with silence, then the words, then trailing silence.
    calibration_frames = settings.vad.calibration_ms // settings.audio.frame_ms
    frames_sequence = [silence_frame] * calibration_frames + [speech_frame] * 10
    frames_sequence += [silence_frame] * 35

    def fake_read(samples):
        if frames_sequence:
            return frames_sequence.pop(0), False
        return silence_frame, False

    stream_mock.read.side_effect = fake_read

    vad_mock = MagicMock()
    vad_mock.is_speech.side_effect = lambda b, sr: b != b"\x00" * len(b)

    out_file = tmp_path / "captured.wav"

    with patch("sounddevice.InputStream") as input_stream_cls, \
         patch("webrtcvad.Vad", return_value=vad_mock):
        input_stream_cls.return_value.__enter__.return_value = stream_mock

        result = transport.record_turn(output_file=out_file)

        assert result.speech_detected is True
        assert result.duration_s > 0
        assert out_file.exists()


def test_audio_transport_record_no_speech(tmp_path: Path) -> None:
    settings = Settings()
    transport = AudioTransport(settings)

    frame_samples = settings.audio.frame_samples
    silence_frame = np.zeros((frame_samples, 1), dtype=np.int16)

    stream_mock = MagicMock()
    stream_mock.read.return_value = (silence_frame, False)

    vad_mock = MagicMock()
    vad_mock.is_speech.return_value = False

    out_file = tmp_path / "silent.wav"

    with patch("sounddevice.InputStream") as input_stream_cls, \
         patch("webrtcvad.Vad", return_value=vad_mock):
        input_stream_cls.return_value.__enter__.return_value = stream_mock

        result = transport.record_turn(output_file=out_file)

        assert result.speech_detected is False
        assert len(result.audio) == 0


def test_audio_transport_adapts_the_gate_to_a_quiet_microphone(tmp_path: Path) -> None:
    """Quiet speech the old fixed 800 gate dropped is now accepted.

    The internal array's words sit far below 800 RMS; the calibrated gate follows
    the room instead, which is what makes one setting work with or without a headset.
    """
    settings = Settings()
    transport = AudioTransport(settings)

    frame_samples = settings.audio.frame_samples
    quiet_speech = np.ones((frame_samples, 1), dtype=np.int16) * 300   # below the old 800 gate
    quiet_room = np.ones((frame_samples, 1), dtype=np.int16) * 20      # a low noise floor

    stream_mock = MagicMock()
    calibration_frames = settings.vad.calibration_ms // settings.audio.frame_ms
    frames_sequence = [quiet_room] * calibration_frames + [quiet_speech] * 10
    frames_sequence += [quiet_room] * 35

    def fake_read(samples):
        if frames_sequence:
            return frames_sequence.pop(0), False
        return quiet_room, False

    stream_mock.read.side_effect = fake_read

    vad_mock = MagicMock()
    vad_mock.is_speech.return_value = True        # VAD alone would accept the loud room too

    out_file = tmp_path / "quiet.wav"

    with patch("sounddevice.InputStream") as input_stream_cls, \
         patch("webrtcvad.Vad", return_value=vad_mock):
        input_stream_cls.return_value.__enter__.return_value = stream_mock

        result = transport.record_turn(output_file=out_file)

    assert result.speech_detected is True          # 300 RMS cleared the calibrated gate
    assert out_file.exists()


def test_audio_transport_play_audio_failure() -> None:
    transport = AudioTransport(Settings())
    with patch("sounddevice.play", side_effect=RuntimeError("Device error")):
        with pytest.raises(AudioError, match="Audio playback failed"):
            transport.play_audio(np.zeros(1600, dtype=np.int16), 16000)
