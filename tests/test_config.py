"""Config defaults, validation, and env overrides.

No mic, no models, no Ollama: these must run anywhere in under a second.
``monkeypatch`` sets env vars and restores them automatically — never set
``$env:BVA_...`` by hand, a leftover variable silently changes behaviour.
"""

from __future__ import annotations

from importlib.metadata import version

import pytest

import bank_voice_assistant
from bank_voice_assistant.config import Settings
from bank_voice_assistant.errors import ConfigError


def test_derived_values_from_defaults() -> None:
    settings = Settings()
    assert settings.audio.frame_samples == 480      # 16 000 Hz * 30 ms
    assert settings.silence_frames_to_stop == 26    # 800 ms // 30 ms
    assert settings.generation.temperature == 0.0


def test_plain_settings_ignores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings() is pure code defaults: env vars must not leak into it."""
    monkeypatch.setenv("BVA_FRAME_MS", "25")
    assert Settings().audio.frame_ms == 30


def test_from_env_applies_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BVA_MIN_SCORE", "0.6")
    monkeypatch.setenv("BVA_TOP_K", "7")
    settings = Settings.from_env()
    assert settings.retrieval.min_score == pytest.approx(0.6)
    assert settings.retrieval.top_k == 7


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("BVA_FRAME_MS", "25"),           # WebRTC VAD allows 10/20/30 only
        ("BVA_SAMPLE_RATE", "12345"),     # not a WebRTC rate
        ("BVA_VAD_AGGRESSIVENESS", "4"),  # 0..3 only
        ("BVA_TOP_K", "0"),               # must be >= 1
        ("BVA_MIN_SCORE", "1.5"),         # must be within 0..1
        ("BVA_TOP_K", "four"),            # not an integer
    ],
)
def test_invalid_values_raise_config_error(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_declared_version_matches_package_metadata() -> None:
    """Guards against pyproject.toml and __init__.__version__ drifting apart."""
    assert version("bank-voice-assistant") == bank_voice_assistant.__version__
