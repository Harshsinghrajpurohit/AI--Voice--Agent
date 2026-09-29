"""CLI smoke tests. No mic, no models, no Ollama."""

from __future__ import annotations

import io

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


def test_console_output_is_lossy_but_never_fatal() -> None:
    """A real KB answer containing '₹' must not abort the turn on a cp1252 console."""
    from bank_voice_assistant.cli import configure_console

    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="")

    configure_console(stdout=stream, stderr=stream)
    try:
        stream.write("Minimum balance is ₹5,000.")
        stream.flush()
    finally:
        stream.detach()

    assert b"5,000" in buffer.getvalue()


def test_configure_console_tolerates_streams_without_reconfigure() -> None:
    from bank_voice_assistant.cli import configure_console

    class PlainStream:
        def write(self, text: str) -> int:
            return len(text)

    configure_console(stdout=PlainStream(), stderr=PlainStream())  # must not raise


def test_voice_flags_registered() -> None:
    from bank_voice_assistant.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["--voice"]).voice is True
    assert parser.parse_args(["--listen"]).voice is True
    assert parser.parse_args([]).voice is False
    assert parser.parse_args([]).max_turns is None
    assert parser.parse_args(["--voice", "--turns", "3"]).max_turns == 3


def test_doctor_reports_audio_devices_and_latency_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from bank_voice_assistant.transport import AudioTransport

    monkeypatch.setattr(
        AudioTransport,
        "describe_devices",
        lambda self: {"input": "Fake Mic (2 ch)", "output": "Fake Speakers (2 ch)"},
    )

    code = main(["--doctor"])
    out = capsys.readouterr().out

    assert code in (EXIT_OK, EXIT_PROBLEM)
    assert "Fake Mic (2 ch)" in out
    assert "latency budget" in out


def test_run_query_prints_the_pipeline_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from bank_voice_assistant.pipeline import TurnResult, TurnTelemetry, VoicePipeline

    def fake_handle_text_turn(self: VoicePipeline, query: str) -> TurnResult:
        return TurnResult(
            query=query,
            answer_text="Home loan interest is 8.75%.",
            is_refusal=False,
            speech_detected=False,
            audio_path=None,
            telemetry=TurnTelemetry(retrieval_s=0.05, llm_s=1.2, chunks_retrieved=2),
        )

    monkeypatch.setattr(VoicePipeline, "handle_text_turn", fake_handle_text_turn)

    code = main(["--ask", "home loan rate"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "Home loan interest is 8.75%." in out
    assert "Chunks retrieved: 2" in out
    assert "Latency:" in out


def test_run_voice_reports_the_session_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from bank_voice_assistant.pipeline import VoicePipeline

    monkeypatch.setattr(VoicePipeline, "run", lambda self, max_turns=None: [])

    code = main(["--voice", "--turns", "1"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "voice assistant ready" in out
    assert "Session ended after 0 answered turn(s)." in out

