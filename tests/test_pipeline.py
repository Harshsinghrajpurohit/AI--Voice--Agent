"""End-to-end orchestration tests (Phase 6).

Every stage is injected as a fake, so these tests never open a microphone, load a
Whisper model, call Ollama, or shell out to Piper. The real ``GuardedGenerator``
and ``GroundedGenerator`` are used where the behaviour under test is the grounding
logic itself — they need no external service when the retrieval result is empty.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from bank_voice_assistant.config import Settings
from bank_voice_assistant.errors import TranscriptionError
from bank_voice_assistant.guardrails import GuardedGenerator
from bank_voice_assistant.kb import Chunk
from bank_voice_assistant.llm import STANDARD_REFUSAL, GenerationResult, GroundedGenerator
from bank_voice_assistant.logging_setup import configure_logging
from bank_voice_assistant.pipeline import (
    NO_SPEECH_MESSAGE,
    TURN_FAILED_MESSAGE,
    TurnResult,
    TurnTelemetry,
    VoicePipeline,
)
from bank_voice_assistant.retrieval import RetrievedChunk
from bank_voice_assistant.transport import AudioCaptureResult


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Drop handlers a previous test may have left pointing at a closed stream."""
    names = ("bva", "bank_voice_assistant")
    for name in names:
        package_logger = logging.getLogger(name)
        package_logger.handlers.clear()
        package_logger.setLevel(logging.NOTSET)
        package_logger.propagate = True
    yield
    for name in names:
        logging.getLogger(name).handlers.clear()


def make_chunk(heading: str = "Home Loans", content: str = "Home loan interest is 8.75%.") -> Chunk:
    return Chunk(
        chunk_id="loans_and_interest_rates#home-loans",
        doc_title="Loans and Interest Rates",
        category="loans",
        heading=heading,
        content=content,
        tags=("home-loan", "interest"),
    )


def make_retrieved(*chunks: Chunk) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=c, score=0.9 - i * 0.1, dense_score=0.8, sparse_score=0.7)
        for i, c in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeRetriever:
    """Records the search arguments the pipeline passed and returns fixed chunks."""

    results: list[RetrievedChunk] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.45, hybrid_alpha: float = 0.5):
        self.calls.append({"query": query, "top_k": top_k, "min_score": min_score, "hybrid_alpha": hybrid_alpha})
        return list(self.results)


@dataclass
class FakeGenerator:
    """Returns a canned result and remembers the queries it was asked."""

    reply: str = "The home loan interest rate is 8.75%."
    is_refusal: bool = False
    queries: list[str] = field(default_factory=list)
    chunks_seen: int = 0

    def generate(self, query: str, chunks: Sequence[RetrievedChunk]) -> GenerationResult:
        self.queries.append(query)
        self.chunks_seen = len(chunks)
        return GenerationResult(
            text=self.reply,
            is_refusal=self.is_refusal,
            word_count=len(self.reply.split()),
            context_chunks=tuple(chunks),
        )


@dataclass
class FakeTranscriber:
    text: str = "What is the home loan interest rate?"
    calls: int = 0

    def transcribe(self, audio: Any) -> str:
        self.calls += 1
        return self.text


@dataclass
class FakeSynthesizer:
    """Writes a small placeholder WAV so the pipeline has a real path for playback.

    Defaults to the system temp directory so a test can never litter the repo.
    """

    target_dir: Path | None = None
    written: list[str] = field(default_factory=list)

    def synthesize(self, text: str, output_path: Path | None = None) -> Path:
        if output_path is not None:
            target = Path(output_path)
        else:
            base = self.target_dir or Path(tempfile.gettempdir())
            target = base / "response.wav"
        target.write_bytes(b"RIFF....WAVE")
        self.written.append(text)
        return target


@dataclass
class FakeTransport:
    """Returns a canned capture and records playback calls."""

    speech: bool = True
    played: list[Any] = field(default_factory=list)
    record_calls: int = 0

    def record_turn(self, output_file: Path | None = None) -> AudioCaptureResult:
        self.record_calls += 1
        audio = np.ones(16_000, dtype=np.int16) * 1200 if self.speech else np.array([], dtype=np.int16)
        return AudioCaptureResult(
            audio=audio,
            sample_rate=16_000,
            duration_s=len(audio) / 16_000,
            max_amplitude=int(np.abs(audio).max()) if len(audio) else 0,
            speech_detected=self.speech,
        )

    def play_audio(self, audio_source: Any, sample_rate: int | None = None) -> None:
        self.played.append(audio_source)


def build_pipeline(**overrides: Any) -> VoicePipeline:
    """Pipeline with deterministic fakes for every external stage."""
    stages: dict[str, Any] = {
        "retriever": FakeRetriever(results=make_retrieved(make_chunk())),
        "generator": FakeGenerator(),
        "transcriber": FakeTranscriber(),
        "synthesizer": FakeSynthesizer(),
        "transport": FakeTransport(),
    }
    stages.update(overrides)
    return VoicePipeline(Settings(), **stages)


# ---------------------------------------------------------------------------
# Text turns
# ---------------------------------------------------------------------------

def test_text_turn_answers_and_reports_telemetry() -> None:
    pipeline = build_pipeline()
    result = pipeline.handle_text_turn("What is the home loan interest rate?")

    assert result.query == "What is the home loan interest rate?"
    assert result.answer_text == "The home loan interest rate is 8.75%."
    assert result.is_refusal is False
    assert result.speech_detected is False
    assert result.audio_path is None
    assert result.telemetry.chunks_retrieved == 1
    assert result.telemetry.total_s >= 0.0
    assert result.telemetry.vad_s == 0.0 and result.telemetry.tts_s == 0.0


def test_text_turn_passes_configured_retrieval_knobs() -> None:
    settings = Settings()
    retriever = FakeRetriever(results=make_retrieved(make_chunk()))
    pipeline = VoicePipeline(settings, retriever=retriever, generator=FakeGenerator())

    pipeline.handle_text_turn("savings account minimum balance")

    assert retriever.calls[0]["query"] == "savings account minimum balance"
    assert retriever.calls[0]["top_k"] == settings.retrieval.top_k
    assert retriever.calls[0]["min_score"] == settings.retrieval.min_score
    assert retriever.calls[0]["hybrid_alpha"] == settings.retrieval.hybrid_alpha


def test_text_turn_without_retrieval_refuses_without_calling_the_llm() -> None:
    """Real guardrails + real grounded generator: empty retrieval must short-circuit."""
    settings = Settings()
    guarded = GuardedGenerator(settings, GroundedGenerator(settings, client=object()))

    pipeline = VoicePipeline(settings, retriever=FakeRetriever(results=[]), generator=guarded)
    result = pipeline.handle_text_turn("What is the CEO's home address?")

    assert result.is_refusal is True
    assert result.answer_text == STANDARD_REFUSAL
    assert result.telemetry.chunks_retrieved == 0



# ---------------------------------------------------------------------------
# Voice turns
# ---------------------------------------------------------------------------

def test_voice_turn_runs_the_full_pipeline(tmp_path: Path) -> None:
    synthesizer = FakeSynthesizer(target_dir=tmp_path)
    transport = FakeTransport()
    transcriber = FakeTranscriber()
    pipeline = build_pipeline(
        synthesizer=synthesizer,
        transport=transport,
        transcriber=transcriber,
    )

    result = pipeline.handle_voice_turn()
    assert result is not None

    assert result.speech_detected is True
    assert result.query == transcriber.text
    assert result.answer_text == "The home loan interest rate is 8.75%."
    assert result.audio_path is not None and result.audio_path.is_file()
    assert synthesizer.written == [result.answer_text]
    assert transport.played == [result.audio_path]
    assert transport.record_calls == 1

    telemetry = result.telemetry
    assert telemetry.audio_s == pytest.approx(1.0)  # 16 000 samples at 16 kHz
    assert telemetry.chunks_retrieved == 1
    assert telemetry.playback_s >= 0.0


def test_voice_turn_skips_stt_and_tts_when_nothing_is_heard() -> None:
    transcriber = FakeTranscriber()
    synthesizer = FakeSynthesizer()
    pipeline = build_pipeline(
        transport=FakeTransport(speech=False),
        transcriber=transcriber,
        synthesizer=synthesizer,
    )

    assert pipeline.handle_voice_turn() is None
    assert transcriber.calls == 0
    assert synthesizer.written == []


def test_voice_turn_skips_answer_when_transcript_is_blank() -> None:
    generator = FakeGenerator()
    synthesizer = FakeSynthesizer()
    pipeline = build_pipeline(
        transcriber=FakeTranscriber(text="   "),
        generator=generator,
        synthesizer=synthesizer,
    )

    assert pipeline.handle_voice_turn() is None
    assert generator.queries == []
    assert synthesizer.written == []


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

class ScriptedPipeline(VoicePipeline):
    """Feeds the loop a scripted sequence of turns.

    Each entry is a question string, ``None`` for a silent turn, or an exception
    to raise — so loop control flow can be tested without any audio hardware.
    """

    def __init__(self, settings: Settings, script: list[Any], **stages: Any) -> None:
        super().__init__(settings, **stages)
        self.script = list(script)
        self.served = 0

    def handle_voice_turn(self, speak_answer: bool = True) -> TurnResult | None:
        self.served += 1
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        if item is None:
            return None
        return TurnResult(
            query=item,
            answer_text="Home loan interest is 8.75%.",
            is_refusal=False,
            speech_detected=True,
            audio_path=None,
            telemetry=TurnTelemetry(stt_s=0.01, retrieval_s=0.02, llm_s=0.03, chunks_retrieved=1),
        )


def build_scripted(script: list[Any], tmp_path: Path) -> tuple[ScriptedPipeline, FakeSynthesizer]:
    synthesizer = FakeSynthesizer(target_dir=tmp_path)
    pipeline = ScriptedPipeline(
        Settings(),
        script,
        synthesizer=synthesizer,
        transport=FakeTransport(),
    )
    return pipeline, synthesizer


def test_run_loop_stops_on_exit_phrase(tmp_path: Path) -> None:
    pipeline, _ = build_scripted(["What is the home loan rate?", "Exit.", "never asked"], tmp_path)

    turns = pipeline.run()

    assert [t.query for t in turns] == ["What is the home loan rate?", "Exit."]
    assert pipeline.served == 2
    assert pipeline.script == ["never asked"]


def test_run_loop_honours_max_turns(tmp_path: Path) -> None:
    pipeline, _ = build_scripted(["one", "two", "three"], tmp_path)

    turns = pipeline.run(max_turns=2)

    assert len(turns) == 2
    assert pipeline.served == 2


def test_run_loop_gives_up_after_repeated_silence(tmp_path: Path) -> None:
    pipeline, synthesizer = build_scripted([None] * 10, tmp_path)

    turns = pipeline.run()

    assert turns == []
    assert pipeline.served == 3  # 1 attempt + settings.pipeline.silent_turn_retries
    assert synthesizer.written == [NO_SPEECH_MESSAGE, NO_SPEECH_MESSAGE]


def test_run_loop_survives_a_failing_turn(tmp_path: Path) -> None:
    pipeline, synthesizer = build_scripted(
        [TranscriptionError("whisper exploded")] * 4, tmp_path
    )

    assert pipeline.run() == []
    assert synthesizer.written == [TURN_FAILED_MESSAGE, TURN_FAILED_MESSAGE]


def test_run_loop_recovers_after_a_failing_turn(tmp_path: Path) -> None:
    pipeline, _ = build_scripted(
        [TranscriptionError("whisper hiccup"), "what is the savings rate?", "quit"], tmp_path
    )

    turns = pipeline.run()

    assert [t.query for t in turns] == ["what is the savings rate?", "quit"]
    assert pipeline.served == 3


def test_voice_turn_can_mute_playback() -> None:
    transport = FakeTransport()
    pipeline = build_pipeline(transport=transport)

    result = pipeline.handle_voice_turn(speak_answer=False)

    assert result is not None
    assert result.audio_path is None
    assert transport.played == []
    assert result.telemetry.tts_s == 0.0


# ---------------------------------------------------------------------------
# Telemetry, logging and wiring
# ---------------------------------------------------------------------------

def test_telemetry_summary_and_budget_flags() -> None:
    telemetry = TurnTelemetry(
        vad_s=0.8,
        stt_s=0.5,
        retrieval_s=0.2,
        llm_s=2.0,
        tts_s=0.6,
        playback_s=3.0,
        audio_s=2.4,
        chunks_retrieved=3,
        is_refusal=False,
    )

    assert telemetry.total_s == pytest.approx(4.1)  # playback excluded on purpose
    assert telemetry.over_target(4.0) is True
    assert telemetry.over_ceiling(8.0) is False

    summary = telemetry.summary()
    assert "VAD 800ms" in summary
    assert "Playback 3000ms" in summary
    assert "Total 4100ms" in summary

    record = telemetry.as_dict(target_s=4.0, ceiling_s=8.0)
    assert record["total_s"] == pytest.approx(4.1)
    assert record["over_target"] is True
    assert record["over_ceiling"] is False
    assert record["chunks_retrieved"] == 3


def test_telemetry_breaches_the_ceiling() -> None:
    telemetry = TurnTelemetry(stt_s=2.0, retrieval_s=1.0, llm_s=5.5, tts_s=1.0)

    assert telemetry.over_ceiling(8.0) is True
    assert telemetry.as_dict(ceiling_s=8.0)["over_ceiling"] is True


def test_configure_logging_emits_module_level_telemetry(capsys: pytest.CaptureFixture[str]) -> None:
    """Module loggers must reach the console, otherwise turn telemetry is invisible."""
    try:
        configure_logging(verbose=False)
        package_logger = logging.getLogger("bank_voice_assistant")
        assert package_logger.handlers
        assert package_logger.level == logging.INFO

        logging.getLogger("bank_voice_assistant.pipeline").info("turn telemetry line")
        assert "turn telemetry line" in capsys.readouterr().err
    finally:
        for name in ("bva", "bank_voice_assistant"):
            logging.getLogger(name).handlers.clear()


def test_pipeline_stages_are_lazy() -> None:
    """Building the pipeline must not load models, open the mic, or need the index."""
    pipeline = VoicePipeline(Settings())

    assert pipeline._retriever is None
    assert pipeline._transcriber is None
    assert pipeline._synthesizer is None
    assert pipeline._transport is None
    assert pipeline._generator is None

