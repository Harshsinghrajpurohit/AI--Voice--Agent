"""Audio-transport helpers: the adaptive speech gate (no microphone involved).

The capture loop needs a real device, so it is exercised live; these tests cover
the pure decision logic it is built from - the part that has to hold on a quiet
internal array and a loud boom headset alike, headphones plugged in or not.
"""

from __future__ import annotations

import numpy as np
import pytest

from bank_voice_assistant.config import VadSettings
from bank_voice_assistant.transport.local_audio import ambient_rms, energy_gate, frame_rms


def test_frame_rms_measures_a_constant_tone_as_its_level() -> None:
    frame = np.full(480, 1000, dtype=np.int16)

    assert frame_rms(frame) == pytest.approx(1000.0)


def test_ambient_rms_is_not_raised_by_a_word_in_the_window() -> None:
    """A talker who starts early must not calibrate the gate to their own voice."""
    quiet = [100.0, 100.0, 100.0, 100.0, 100.0]
    with_a_word = quiet + [5000.0]

    assert ambient_rms(quiet) == pytest.approx(100.0)
    assert ambient_rms(with_a_word) == pytest.approx(100.0)   # mean would be 916


def test_ambient_rms_of_an_empty_window_is_zero() -> None:
    assert ambient_rms([]) == 0.0


def test_energy_gate_tracks_noise_between_a_floor_and_the_ceiling() -> None:
    vad = VadSettings()                                        # floor 120, x2.0, ceiling 800

    assert energy_gate(0.0, vad) == pytest.approx(120.0)       # dead-quiet mic -> floor
    assert energy_gate(50.0, vad) == pytest.approx(120.0)
    assert energy_gate(200.0, vad) == pytest.approx(400.0)     # 200 x 2.0
    assert energy_gate(10_000.0, vad) == pytest.approx(800.0)  # never above the ceiling


def test_a_disabled_calibration_pins_the_fixed_gate() -> None:
    """``calibration_ms = 0`` is the escape hatch back to one fixed threshold."""
    vad = VadSettings(calibration_ms=0, energy_threshold=333)

    assert energy_gate(5.0, vad) == pytest.approx(333.0)
    assert energy_gate(9_999.0, vad) == pytest.approx(333.0)


def test_a_quiet_mic_and_a_loud_mic_both_admit_speech() -> None:
    """The whole point: one setting clears speech on a quiet array and a loud headset."""
    vad = VadSettings()

    quiet_array_speech = 700.0        # internal array: noise 10, speech ~700
    loud_headset_speech = 3000.0      # headset: noise 5, speech ~3000

    assert quiet_array_speech > energy_gate(10.0, vad)
    assert loud_headset_speech > energy_gate(5.0, vad)
