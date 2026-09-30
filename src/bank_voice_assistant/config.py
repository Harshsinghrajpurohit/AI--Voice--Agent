"""Single source of truth for runtime configuration.

Nothing outside this module may hardcode a path, model name, or threshold:
defaults live here and every value can be overridden with a ``BVA_``-prefixed
environment variable, so tests and future transports (web/SIP) reuse the exact
same pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""Repo root for an editable install (``<root>/src/bank_voice_assistant/config.py``)."""


# ---------------------------------------------------------------------------
# Environment helpers. Every bad value becomes a ConfigError with the variable
# name in it, so the CLI can print something actionable instead of a traceback.
# ---------------------------------------------------------------------------
def _raw(name: str) -> str | None:
    return os.environ.get(f"BVA_{name}", "").strip() or None


def _str(name: str, default: str) -> str:
    return _raw(name) or default


def _int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"BVA_{name}={raw!r} must be an integer") from exc


def _float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"BVA_{name}={raw!r} must be a number") from exc


def _path(name: str, default: Path) -> Path:
    raw = _raw(name)
    return Path(raw).expanduser().resolve() if raw else default


def _str_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated phrase list; blank entries are dropped, empties rejected."""
    raw = _raw(name)
    if raw is None:
        return default
    items = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not items:
        raise ConfigError(f"BVA_{name} must list at least one comma-separated value")
    return items


@dataclass(frozen=True, slots=True)
class Paths:
    """Filesystem layout. Config describes paths, it never creates them."""

    project_root: Path = PROJECT_ROOT
    kb_dir: Path = PROJECT_ROOT / "data" / "kb"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"
    index_dir: Path = PROJECT_ROOT / ".cache" / "index"
    audio_in: Path = PROJECT_ROOT / "audio.wav"
    audio_out: Path = PROJECT_ROOT / "response.wav"
    voice_model: Path = PROJECT_ROOT / "en_US-lessac-medium.onnx"

    @classmethod
    def from_env(cls) -> Paths:
        base = cls()
        return cls(
            kb_dir=_path("KB_DIR", base.kb_dir),
            eval_dir=_path("EVAL_DIR", base.eval_dir),
            index_dir=_path("INDEX_DIR", base.index_dir),
            voice_model=_path("VOICE_MODEL", base.voice_model),
        )


@dataclass(frozen=True, slots=True)
class AudioSettings:
    """Mic/speaker format. These three values are constrained by WebRTC VAD."""

    sample_rate: int = 16_000
    frame_ms: int = 30
    channels: int = 1
    dtype: str = "int16"

    def __post_init__(self) -> None:
        if self.sample_rate not in (8_000, 16_000, 32_000, 48_000):
            raise ConfigError("sample_rate must be 8000, 16000, 32000 or 48000 (WebRTC VAD)")
        if self.frame_ms not in (10, 20, 30):
            raise ConfigError("frame_ms must be 10, 20 or 30 (WebRTC VAD)")
        if self.channels != 1:
            raise ConfigError("the VAD/Whisper pipeline is mono only")

    @property
    def frame_samples(self) -> int:
        """Samples per VAD frame: 480 at 16 kHz / 30 ms."""
        return self.sample_rate * self.frame_ms // 1000


@dataclass(frozen=True, slots=True)
class VadSettings:
    """Endpointing tunables — the latency-vs-interruption dial."""

    aggressiveness: int = 3
    energy_threshold: int = 800
    silence_ms_to_stop: int = 800
    max_record_seconds: int = 15
    min_speech_frames: int = 5

    def __post_init__(self) -> None:
        if not 0 <= self.aggressiveness <= 3:
            raise ConfigError("aggressiveness must be 0..3 (0 = keep most audio, 3 = filter most)")
        if self.energy_threshold < 0:
            raise ConfigError("energy_threshold must be >= 0 (int16 RMS gate)")
        if self.silence_ms_to_stop <= 0:
            raise ConfigError("silence_ms_to_stop must be > 0")
        if not 1 <= self.max_record_seconds <= 120:
            raise ConfigError("max_record_seconds must be 1..120")
        if self.min_speech_frames < 1:
            raise ConfigError("min_speech_frames must be >= 1")


@dataclass(frozen=True, slots=True)
class SttSettings:
    """Faster-Whisper settings. CPU int8 keeps the 4 GB GPU free for the LLM."""

    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    initial_prompt: str = (
        "Banking FAQ about savings and current accounts, fixed deposits, credit cards, "
        "loans, interest rates, fees and charges, and KYC documents."
    )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """Hybrid retrieval: dense (meaning) + BM25 (exact digits and product names)."""

    embed_model: str = "BAAI/bge-small-en-v1.5"
    candidate_pool: int = 20
    top_k: int = 4
    min_score: float = 0.45
    hybrid_alpha: float = 0.5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigError("top_k must be >= 1")
        if self.candidate_pool < self.top_k:
            raise ConfigError("candidate_pool must be >= top_k")
        if not 0.0 <= self.min_score <= 1.0:
            raise ConfigError("min_score must be within 0.0..1.0")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ConfigError("hybrid_alpha must be within 0.0..1.0 (1.0 = dense only)")


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Answer generation. The grounding rules are encoded here, not left to habit."""

    model: str = "llama3.2"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    num_ctx: int = 4096
    keep_alive: str = "10m"
    request_timeout_s: float = 60.0
    max_words: int = 35
    max_sentences: int = 3
    numeric_retry_attempts: int = 1

    def __post_init__(self) -> None:
        if self.temperature != 0.0:
            raise ConfigError("temperature must stay 0.0 — this project does not sample")
        if self.max_words < 1 or self.max_sentences < 1:
            raise ConfigError("max_words and max_sentences must be >= 1")
        if self.numeric_retry_attempts < 0:
            raise ConfigError("numeric_retry_attempts must be >= 0")


@dataclass(frozen=True, slots=True)
class TtsSettings:
    """Piper playback. length_scale > 1.0 speaks slower."""

    piper_exe: str = "piper"
    length_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.length_scale <= 2.0:
            raise ConfigError("length_scale must be within 0.5..2.0")


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """Interactive loop behaviour and the latency budget the project promises.

    ``latency_target_s`` is what the turn telemetry is measured against;
    ``latency_ceiling_s`` is the point where a turn is logged as a hard breach.
    """

    latency_target_s: float = 4.0
    latency_ceiling_s: float = 8.0
    max_turns: int = 0
    silent_turn_retries: int = 2
    exit_phrases: tuple[str, ...] = ("exit", "quit", "stop", "goodbye")

    def __post_init__(self) -> None:
        if self.latency_target_s <= 0.0:
            raise ConfigError("latency_target_s must be > 0")
        if self.latency_ceiling_s < self.latency_target_s:
            raise ConfigError("latency_ceiling_s must be >= latency_target_s")
        if self.max_turns < 0:
            raise ConfigError("max_turns must be >= 0 (0 = run until the caller stops)")
        if self.silent_turn_retries < 0:
            raise ConfigError("silent_turn_retries must be >= 0")
        if not self.exit_phrases:
            raise ConfigError("exit_phrases must not be empty")


@dataclass(frozen=True, slots=True)
class EvalSettings:
    """Pass/fail floors for the Phase 7 evaluation harness.

    Each default cites its source: the project's grounding promise, the Phase 7
    exit criterion in ``Phases.md``, or the latency budget in
    :class:`PipelineSettings`. These bound a *run*, they do not describe today's
    scores, so a breached threshold means the assistant changed.
    """

    min_grounding_rate: float = 1.0
    """Every number in an answer must come from a retrieved record."""

    min_answer_accuracy: float = 0.90
    min_fact_recall: float = 0.90
    min_retrieval_hit_rate: float = 0.85
    """Diagnostic floor for tuning ``top_k`` / ``min_score``."""

    min_refusal_recall: float = 1.0
    """Every refusal row must be declined. The floor was 0.95 while
    ``adv-dev-mode-01`` was a documented known failure; Step 7.6 refuses
    instruction-override attempts before retrieval, so the exemption is gone."""

    min_safe_decline_rate: float = 0.90
    max_over_refusal_rate: float = 0.10
    """Refusing a question the KB can answer is the costly mistake."""

    max_p50_latency_s: float = 4.0
    """Phase 7 exit criterion: under 4.0s per response."""

    max_p95_latency_s: float = 8.0
    """Equal to ``PipelineSettings.latency_ceiling_s``: a turn above it is a breach."""

    def __post_init__(self) -> None:
        rates = {
            "min_grounding_rate": self.min_grounding_rate,
            "min_answer_accuracy": self.min_answer_accuracy,
            "min_fact_recall": self.min_fact_recall,
            "min_retrieval_hit_rate": self.min_retrieval_hit_rate,
            "min_refusal_recall": self.min_refusal_recall,
            "min_safe_decline_rate": self.min_safe_decline_rate,
            "max_over_refusal_rate": self.max_over_refusal_rate,
        }
        for name, value in rates.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be within 0.0..1.0 (it is a rate)")
        if self.max_p50_latency_s <= 0.0 or self.max_p95_latency_s <= 0.0:
            raise ConfigError("max_p50_latency_s and max_p95_latency_s must be > 0")
        if self.max_p95_latency_s < self.max_p50_latency_s:
            raise ConfigError("max_p95_latency_s must be >= max_p50_latency_s")


@dataclass(frozen=True, slots=True)
class Settings:
    """Aggregate settings object. Build it with ``Settings.from_env()``."""

    paths: Paths = field(default_factory=Paths)
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    stt: SttSettings = field(default_factory=SttSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)

    @classmethod
    def from_env(cls) -> Settings:
        """Code defaults, overridden by any ``BVA_*`` variables that are set."""
        base = cls()
        return cls(
            paths=Paths.from_env(),
            audio=AudioSettings(
                sample_rate=_int("SAMPLE_RATE", base.audio.sample_rate),
                frame_ms=_int("FRAME_MS", base.audio.frame_ms),
            ),
            vad=VadSettings(
                aggressiveness=_int("VAD_AGGRESSIVENESS", base.vad.aggressiveness),
                energy_threshold=_int("ENERGY_THRESHOLD", base.vad.energy_threshold),
                silence_ms_to_stop=_int("SILENCE_MS_TO_STOP", base.vad.silence_ms_to_stop),
                max_record_seconds=_int("MAX_RECORD_SECONDS", base.vad.max_record_seconds),
                min_speech_frames=_int("MIN_SPEECH_FRAMES", base.vad.min_speech_frames),
            ),
            stt=SttSettings(
                model_size=_str("STT_MODEL", base.stt.model_size),
                device=_str("STT_DEVICE", base.stt.device),
                compute_type=_str("STT_COMPUTE_TYPE", base.stt.compute_type),
                language=_str("STT_LANGUAGE", base.stt.language),
            ),
            retrieval=RetrievalSettings(
                embed_model=_str("EMBED_MODEL", base.retrieval.embed_model),
                top_k=_int("TOP_K", base.retrieval.top_k),
                min_score=_float("MIN_SCORE", base.retrieval.min_score),
                hybrid_alpha=_float("HYBRID_ALPHA", base.retrieval.hybrid_alpha),
            ),
            generation=GenerationSettings(
                model=_str("LLM_MODEL", base.generation.model),
                host=_str("OLLAMA_HOST", base.generation.host),
                max_words=_int("MAX_WORDS", base.generation.max_words),
                max_sentences=_int("MAX_SENTENCES", base.generation.max_sentences),
            ),
            tts=TtsSettings(
                piper_exe=_str("PIPER_EXE", base.tts.piper_exe),
                length_scale=_float("LENGTH_SCALE", base.tts.length_scale),
            ),
            pipeline=PipelineSettings(
                latency_target_s=_float("LATENCY_TARGET_S", base.pipeline.latency_target_s),
                latency_ceiling_s=_float("LATENCY_CEILING_S", base.pipeline.latency_ceiling_s),
                max_turns=_int("MAX_TURNS", base.pipeline.max_turns),
                silent_turn_retries=_int("SILENT_TURN_RETRIES", base.pipeline.silent_turn_retries),
                exit_phrases=_str_tuple("EXIT_PHRASES", base.pipeline.exit_phrases),
            ),
            eval=EvalSettings(
                min_grounding_rate=_float("MIN_GROUNDING_RATE", base.eval.min_grounding_rate),
                min_answer_accuracy=_float("MIN_ANSWER_ACCURACY", base.eval.min_answer_accuracy),
                min_fact_recall=_float("MIN_FACT_RECALL", base.eval.min_fact_recall),
                min_retrieval_hit_rate=_float(
                    "MIN_RETRIEVAL_HIT_RATE", base.eval.min_retrieval_hit_rate
                ),
                min_refusal_recall=_float("MIN_REFUSAL_RECALL", base.eval.min_refusal_recall),
                min_safe_decline_rate=_float(
                    "MIN_SAFE_DECLINE_RATE", base.eval.min_safe_decline_rate
                ),
                max_over_refusal_rate=_float(
                    "MAX_OVER_REFUSAL_RATE", base.eval.max_over_refusal_rate
                ),
                max_p50_latency_s=_float("MAX_P50_LATENCY_S", base.eval.max_p50_latency_s),
                max_p95_latency_s=_float("MAX_P95_LATENCY_S", base.eval.max_p95_latency_s),
            ),
        )

    @property
    def silence_frames_to_stop(self) -> int:
        """Consecutive silent frames that mean "the caller stopped talking"."""
        return max(1, self.vad.silence_ms_to_stop // self.audio.frame_ms)

    def as_log_dict(self) -> dict[str, object]:
        """Flat view for ``--verbose`` startup logging and eval run records."""
        return {
            "llm_model": self.generation.model,
            "stt_model": self.stt.model_size,
            "embed_model": self.retrieval.embed_model,
            "top_k": self.retrieval.top_k,
            "min_score": self.retrieval.min_score,
            "hybrid_alpha": self.retrieval.hybrid_alpha,
            "vad_aggressiveness": self.vad.aggressiveness,
            "frame_samples": self.audio.frame_samples,
            "silence_frames_to_stop": self.silence_frames_to_stop,
            "max_words": self.generation.max_words,
            "latency_target_s": self.pipeline.latency_target_s,
            "kb_dir": str(self.paths.kb_dir),
            "index_dir": str(self.paths.index_dir),
        }
