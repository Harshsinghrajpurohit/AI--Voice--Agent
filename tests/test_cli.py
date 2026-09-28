"""CLI smoke tests. No mic, no models, no Ollama."""

from __future__ import annotations

import pytest

from bank_voice_assistant import __version__
from bank_voice_assistant.cli import EXIT_OK, EXIT_PROBLEM, EXIT_USAGE, main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == EXIT_OK
    assert __version__ in capsys.readouterr().out


def test_verbose_and_quiet_conflict() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["-v", "-q"])
    assert excinfo.value.code == EXIT_USAGE


def test_unknown_flag_is_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--nope"])
    assert excinfo.value.code == EXIT_USAGE


def test_doctor_never_crashes(capsys: pytest.CaptureFixture[str]) -> None:
    """Doctor may legitimately FAIL on a bare machine; it must never raise."""
    code = main(["--doctor"])
    assert code in (EXIT_OK, EXIT_PROBLEM)
    assert "Environment check" in capsys.readouterr().out


def test_query_flag_registered() -> None:
    from bank_voice_assistant.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["--query", "What are your savings account rates?"])
    assert args.query == "What are your savings account rates?"

