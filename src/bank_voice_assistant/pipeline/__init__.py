"""End-to-end orchestration: one turn = capture -> transcribe -> answer -> speak.

Everything the assistant needs is assembled here, and every stage is injectable,
so the whole loop can be exercised in tests without a microphone, a Whisper model,
an Ollama server, or a Piper binary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..errors import BankVoiceAssistantError
from ..guardrails import GuardedGenerator, detect_instruction_override
from ..llm import GenerationResult, GroundedGenerator, refusal_result
from ..retrieval import HybridRetriever
from ..stt import Transcriber
from ..transport import AudioCaptureResult, AudioTransport
from ..tts import PiperTTS

logger = logging.getLogger(__name__)

NO_SPEECH_MESSAGE = "I did not catch that. Please ask your question again."
TURN_FAILED_MESSAGE = "Sorry, I could not process that. Please try again."


@dataclass(frozen=True, slots=True)
class TurnTelemetry:
    """Per-stage wall-clock breakdown of a single conversational turn."""

    vad_s: float = 0.0
    stt_s: float = 0.0
    retrieval_s: float = 0.0
    llm_s: float = 0.0
    tts_s: float = 0.0
    playback_s: float = 0.0
    audio_s: float = 0.0
    chunks_retrieved: int = 0
    is_refusal: bool = False

    @property
    def total_s(self) -> float:
        """End-of-speech -> reply ready. Playback is excluded from the budget."""
        return self.vad_s + self.stt_s + self.retrieval_s + self.llm_s + self.tts_s

    def over_target(self, target_s: float) -> bool:
        """True when the turn missed the latency target."""
        return self.total_s > target_s

    def over_ceiling(self, ceiling_s: float) -> bool:
        """True when the turn breached the hard latency ceiling."""
        return self.total_s > ceiling_s

    def summary(self) -> str:
        """One-line breakdown for the console and for Phase 7 run records."""
        return (
            f"VAD {self.vad_s * 1000:.0f}ms | STT {self.stt_s * 1000:.0f}ms | "
            f"Retrieval {self.retrieval_s * 1000:.0f}ms | LLM {self.llm_s * 1000:.0f}ms | "
            f"TTS {self.tts_s * 1000:.0f}ms | Playback {self.playback_s * 1000:.0f}ms | "
            f"Total {self.total_s * 1000:.0f}ms"
        )

    def as_dict(self, target_s: float = 0.0, ceiling_s: float = 0.0) -> dict[str, object]:
        """Flat record for logs and the Phase 7 evaluation runner."""
        return {
            "vad_s": round(self.vad_s, 4),
            "stt_s": round(self.stt_s, 4),
            "retrieval_s": round(self.retrieval_s, 4),
            "llm_s": round(self.llm_s, 4),
            "tts_s": round(self.tts_s, 4),
            "playback_s": round(self.playback_s, 4),
            "audio_s": round(self.audio_s, 4),
            "total_s": round(self.total_s, 4),
            "chunks_retrieved": self.chunks_retrieved,
            "is_refusal": self.is_refusal,
            "over_target": self.over_target(target_s) if target_s else False,
            "over_ceiling": self.over_ceiling(ceiling_s) if ceiling_s else False,
        }


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What one handled turn produced."""

    query: str
    answer_text: str
    is_refusal: bool
    speech_detected: bool
    audio_path: Path | None
    telemetry: TurnTelemetry


class VoicePipeline:
    """Assembles STT, retrieval, guardrailed generation, TTS and audio I/O.

    Every collaborator is optional and built lazily on first use, so importing
    this module never loads a model and tests can inject fakes for each stage.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        retriever: HybridRetriever | None = None,
        transcriber: Transcriber | None = None,
        synthesizer: PiperTTS | None = None,
        transport: AudioTransport | None = None,
        generator: GuardedGenerator | None = None,
    ) -> None:
        self.settings = settings
        self._retriever = retriever
        self._transcriber = transcriber
        self._synthesizer = synthesizer
        self._transport = transport
        self._generator = generator

    # -- lazily constructed stages ------------------------------------------
    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever.load(
                index_dir=self.settings.paths.index_dir,
                embed_model=self.settings.retrieval.embed_model,
            )
        return self._retriever

    @property
    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = Transcriber(self.settings)
        return self._transcriber

    @property
    def synthesizer(self) -> PiperTTS:
        if self._synthesizer is None:
            self._synthesizer = PiperTTS(self.settings)
        return self._synthesizer

    @property
    def transport(self) -> AudioTransport:
        if self._transport is None:
            self._transport = AudioTransport(self.settings)
        return self._transport

    @property
    def generator(self) -> GuardedGenerator:
        if self._generator is None:
            self._generator = GuardedGenerator(self.settings, GroundedGenerator(self.settings))
        return self._generator

    # -- stages -------------------------------------------------------------
    def answer(self, query: str) -> tuple[GenerationResult, float, float]:
        """Retrieve and generate. Returns ``(result, retrieval_s, llm_s)``.

        An instruction-override attempt is refused before retrieval: the input is
        not a question about the bank, so there is nothing to look up, and the
        model is never given the chance to act on the instruction.
        """
        override = detect_instruction_override(query)
        if override is not None:
            logger.warning("Instruction-override attempt refused before retrieval: %s", override)
            return refusal_result(), 0.0, 0.0

        t0 = time.perf_counter()
        chunks = self.retriever.search(
            query=query,
            top_k=self.settings.retrieval.top_k,
            min_score=self.settings.retrieval.min_score,
            hybrid_alpha=self.settings.retrieval.hybrid_alpha,
        )
        retrieval_s = time.perf_counter() - t0
        logger.debug("Retrieved %d chunk(s) for %r", len(chunks), query)

        t1 = time.perf_counter()
        result = self.generator.generate(query, chunks)
        llm_s = time.perf_counter() - t1
        return result, retrieval_s, llm_s

    def speak(self, text: str) -> tuple[Path, float, float]:
        """Synthesize and play ``text``. Returns ``(audio_path, tts_s, playback_s)``."""
        t0 = time.perf_counter()
        audio_path = self.synthesizer.synthesize(text)
        tts_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.transport.play_audio(audio_path)
        playback_s = time.perf_counter() - t1
        return audio_path, tts_s, playback_s

    # -- turns --------------------------------------------------------------
    def handle_text_turn(self, query: str) -> TurnResult:
        """Answer a typed question: retrieval + generation only, no audio."""
        result, retrieval_s, llm_s = self.answer(query)
        telemetry = TurnTelemetry(
            retrieval_s=retrieval_s,
            llm_s=llm_s,
            chunks_retrieved=len(result.context_chunks),
            is_refusal=result.is_refusal,
        )
        self._log_turn(query, telemetry)
        return TurnResult(
            query=query,
            answer_text=result.text,
            is_refusal=result.is_refusal,
            speech_detected=False,
            audio_path=None,
            telemetry=telemetry,
        )

    def handle_voice_turn(self, speak_answer: bool = True) -> TurnResult | None:
        """Run one full spoken turn.

        Returns ``None`` when nothing intelligible was captured, which tells the
        caller to prompt again rather than to treat it as an answer.
        """
        t0 = time.perf_counter()
        capture: AudioCaptureResult = self.transport.record_turn()
        vad_s = time.perf_counter() - t0

        if not capture.speech_detected or capture.audio.size == 0:
            logger.info("No speech detected in this turn.")
            return None

        t1 = time.perf_counter()
        query = self.transcriber.transcribe(capture.audio)
        stt_s = time.perf_counter() - t1

        if not query.strip():
            logger.info("Speech detected but nothing was transcribed.")
            return None

        logger.info("Caller: %s", query)
        result, retrieval_s, llm_s = self.answer(query)

        audio_path: Path | None = None
        tts_s = 0.0
        playback_s = 0.0
        if speak_answer:
            audio_path, tts_s, playback_s = self.speak(result.text)

        telemetry = TurnTelemetry(
            vad_s=vad_s,
            stt_s=stt_s,
            retrieval_s=retrieval_s,
            llm_s=llm_s,
            tts_s=tts_s,
            playback_s=playback_s,
            audio_s=capture.duration_s,
            chunks_retrieved=len(result.context_chunks),
            is_refusal=result.is_refusal,
        )
        self._log_turn(query, telemetry)
        return TurnResult(
            query=query,
            answer_text=result.text,
            is_refusal=result.is_refusal,
            speech_detected=True,
            audio_path=audio_path,
            telemetry=telemetry,
        )

    # -- interactive loop ---------------------------------------------------
    def run(self, max_turns: int | None = None) -> list[TurnResult]:
        """Listen, answer and speak until an exit phrase, ``max_turns`` or Ctrl+C.

        A turn that captures nothing or fails is retried at most
        ``settings.pipeline.silent_turn_retries`` times in a row, so an unattended
        session cannot spin forever.
        """
        cfg = self.settings.pipeline
        limit = cfg.max_turns if max_turns is None else max_turns
        turns: list[TurnResult] = []
        failed_streak = 0

        logger.info("Listening — say %r to stop.", cfg.exit_phrases[0])

        while limit == 0 or len(turns) < limit:
            try:
                turn = self.handle_voice_turn()
            except KeyboardInterrupt:
                logger.info("Interrupted — ending the session.")
                break
            except BankVoiceAssistantError as exc:
                failed_streak += 1
                logger.error("%s", exc)
                if failed_streak > cfg.silent_turn_retries:
                    logger.error("Too many failed turns in a row — ending the session.")
                    break
                self._say_safely(TURN_FAILED_MESSAGE)
                continue

            if turn is None:
                failed_streak += 1
                if failed_streak > cfg.silent_turn_retries:
                    logger.info("No speech detected %d times — ending the session.", failed_streak)
                    break
                self._say_safely(NO_SPEECH_MESSAGE)
                continue

            failed_streak = 0
            turns.append(turn)

            if self._is_exit_phrase(turn.query):
                logger.info("Exit phrase heard — ending the session.")
                break

        logger.info("Session finished: %d answered turn(s).", len(turns))
        return turns

    # -- helpers ------------------------------------------------------------
    def _is_exit_phrase(self, text: str) -> bool:
        normalized = text.strip().lower().strip(" .!?,")
        return normalized in self.settings.pipeline.exit_phrases

    def _say_safely(self, text: str) -> None:
        """Speak a courtesy line; never let a TTS/audio failure break the loop."""
        try:
            self.speak(text)
        except BankVoiceAssistantError as exc:
            logger.debug("Could not speak the courtesy line: %s", exc)

    def _log_turn(self, query: str, telemetry: TurnTelemetry) -> None:
        cfg = self.settings.pipeline
        logger.info(
            "[turn] %s | refusal=%s | query=%r",
            telemetry.summary(),
            telemetry.is_refusal,
            query,
        )
        if telemetry.over_ceiling(cfg.latency_ceiling_s):
            logger.warning(
                "[turn] latency ceiling breached: %.2fs > %.2fs",
                telemetry.total_s,
                cfg.latency_ceiling_s,
            )
        elif telemetry.over_target(cfg.latency_target_s):
            logger.info(
                "[turn] above latency target: %.2fs > %.2fs",
                telemetry.total_s,
                cfg.latency_target_s,
            )


__all__ = ["TurnResult", "TurnTelemetry", "VoicePipeline"]

