"""Command-line entry point.

Modes: ``--ask "question"`` (one-shot text turn), ``--voice`` (interactive audio
loop), ``--build-index`` (rebuild the hybrid index), ``--eval`` (score the golden
dataset), ``--doctor`` (offline environment self-check). Everything heavy (models,
index, mic) is imported and loaded lazily by the stage that needs it, so ``--help``
and ``--doctor`` stay fast.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import __version__
from .config import Settings
from .errors import BankVoiceAssistantError
from .logging_setup import configure_logging

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_USAGE = 2
EXIT_GATE_FAILED = 3
"""The evaluation ran and missed a threshold: a quality verdict, not a crash."""


def configure_console(stdout: TextIO | None = None, stderr: TextIO | None = None) -> None:
    """Make console output lossy instead of fatal.

    Knowledge base answers legitimately contain characters such as ``₹`` that a
    legacy Windows code page cannot encode. Without this, printing a perfectly
    valid answer raised ``UnicodeEncodeError`` and aborted the turn.
    """
    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except Exception:  # exotic stream (pytest capture, redirected pipe): not fatal
            continue


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (kept separate so tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="bank-voice",
        description="Fully local, knowledge-base-grounded voice assistant for banking FAQs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="debug output")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    parser.add_argument("--doctor", action="store_true", help="check the local environment, then exit")
    parser.add_argument("--build-index", action="store_true", help="build hybrid retrieval index from KB files")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="score the golden dataset against the live assistant and print the report",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="with --eval: also write the full run record to PATH as JSON",
    )
    parser.add_argument(
        "--voice", "--listen",
        dest="voice",
        action="store_true",
        help="start the interactive voice loop (mic -> answer -> speakers)",
    )
    parser.add_argument(
        "--turns",
        dest="max_turns",
        type=int,
        default=None,
        metavar="N",
        help="stop the voice loop after N answered turns (default: until you say 'exit')",
    )
    parser.add_argument(
        "-a", "--ask", "--query",
        dest="query",
        type=str,
        help="ask a banking question to test retrieval and guardrailed generation",
    )

    return parser


def run_doctor(settings: Settings) -> int:
    """Offline self-check of everything this phase can verify. Returns an exit code."""
    print("Environment check")
    failures = 0

    print(
        f"[OK  ] config            {settings.generation.model} · "
        f"{settings.stt.model_size} · {settings.retrieval.embed_model}"
    )

    piper = shutil.which(settings.tts.piper_exe)
    if piper:
        print(f"[OK  ] piper            {piper}")
    else:
        print("[FAIL] piper            not on PATH — run inside the venv (piper-tts installed)")
        failures += 1

    if settings.paths.voice_model.is_file():
        print(f"[OK  ] voice model      {settings.paths.voice_model.name}")
    else:
        print(f"[FAIL] voice model      missing: {settings.paths.voice_model}")
        failures += 1

    kb_files = sorted(settings.paths.kb_dir.glob("*.md")) if settings.paths.kb_dir.is_dir() else []
    if kb_files:
        print(f"[OK  ] knowledge base   {len(kb_files)} file(s) in {settings.paths.kb_dir}")
    else:
        print(f"[FAIL] knowledge base   no markdown documents in {settings.paths.kb_dir}")
        failures += 1

    index_files = ("chunks.json", "embeddings.npy", "bm25.json")
    if (settings.paths.index_dir / index_files[0]).is_file() and all(
        (settings.paths.index_dir / name).is_file() for name in index_files[1:]
    ):
        print(f"[OK  ] retrieval index  {settings.paths.index_dir}")
    else:
        print(
            f"[PEND] retrieval index  not built — run: bank-voice --build-index "
            f"({settings.paths.index_dir})"
        )

    try:
        from .transport import AudioTransport

        devices = AudioTransport(settings).describe_devices()
        print(f"[OK  ] audio devices    in: {devices['input']} | out: {devices['output']}")
    except BankVoiceAssistantError as exc:
        print(f"[FAIL] audio devices    {exc}")
        failures += 1
    except Exception as exc:  # defensive: --doctor must never crash
        print(f"[FAIL] audio devices    unexpected error: {exc}")
        failures += 1

    cfg_pipeline = settings.pipeline
    print(
        f"[OK  ] latency budget   target {cfg_pipeline.latency_target_s:.1f}s · "
        f"ceiling {cfg_pipeline.latency_ceiling_s:.1f}s (VAD + STT + retrieval + LLM + TTS)"
    )

    print()
    if failures:
        print(f"{failures} problem(s) found.")
        return EXIT_PROBLEM
    print("All current-phase checks passed.")
    return EXIT_OK


def run_build_index(settings: Settings) -> int:
    """Build dense + BM25 index from knowledge base files."""
    from .kb import KnowledgeBase
    from .retrieval import HybridRetriever

    print(f"Loading knowledge base from {settings.paths.kb_dir}...")
    kb = KnowledgeBase(settings.paths.kb_dir)
    chunks = kb.load_all_chunks()
    print(f"Loaded {len(chunks)} chunks. Building hybrid index (dense + BM25)...")

    HybridRetriever.build_and_save(
        chunks=chunks,
        output_dir=settings.paths.index_dir,
        embed_model=settings.retrieval.embed_model,
    )
    print(f"[OK  ] Hybrid index successfully built in {settings.paths.index_dir}")
    return EXIT_OK


def run_eval(settings: Settings, report: Path | None = None) -> int:
    """Score the golden dataset and print the report. Returns an exit code.

    A missed threshold is a result rather than a failure to run, so it gets its
    own exit code: a script can then tell "the assistant answered badly" apart
    from "the evaluation never happened".
    """
    from .eval import run_evaluation, write_report

    run = run_evaluation(settings)
    for line in run.summary_lines():
        print(line)

    if report is not None:
        write_report(run, report)
        print(f"[OK  ] run record written to {report}")

    return EXIT_OK if run.passed else EXIT_GATE_FAILED


def run_query(settings: Settings, query: str) -> int:
    """One-shot text turn: retrieval + guardrailed generation, no audio."""
    from .pipeline import VoicePipeline

    result = VoicePipeline(settings).handle_text_turn(query)

    print(f"\nQuery: {query}")
    print(f"Refusal: {result.is_refusal}")
    print(f"Chunks retrieved: {result.telemetry.chunks_retrieved}")
    print(f"Answer: {result.answer_text}")
    print(f"Latency: {result.telemetry.summary()}\n")
    return EXIT_OK


def run_voice(settings: Settings, max_turns: int | None = None) -> int:
    """Interactive loop: mic -> Whisper -> retrieval -> LLM -> Piper -> speakers."""
    from .pipeline import VoicePipeline

    pipeline = VoicePipeline(settings)
    print("Northwind Bank voice assistant ready.")
    print(f"Speak your question; say '{settings.pipeline.exit_phrases[0]}' to stop.")

    try:
        turns = pipeline.run(max_turns=max_turns)
    except KeyboardInterrupt:
        print("\nStopped.")
        return EXIT_OK

    print(f"\nSession ended after {len(turns)} answered turn(s).")
    return EXIT_OK




def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for both ``bank-voice`` and ``python -m bank_voice_assistant``."""
    args = build_parser().parse_args(argv)
    configure_console()

    if args.report is not None and not args.eval:
        print("--report writes an evaluation run record, so it needs --eval.", file=sys.stderr)
        return EXIT_USAGE

    logger = configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        settings = Settings.from_env()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("effective settings: %s", settings.as_log_dict())

        if args.doctor:
            return run_doctor(settings)

        if args.build_index:
            return run_build_index(settings)

        if args.eval:
            return run_eval(settings, report=args.report)

        if args.query:
            return run_query(settings, args.query)

        if args.voice:
            return run_voice(settings, max_turns=args.max_turns)

        print(
            "Nothing to do. Try --ask \"what is the home loan interest rate?\", "
            "--voice, --build-index, --eval or --doctor."
        )
        return EXIT_OK
    except BankVoiceAssistantError as exc:
        # Project errors carry actionable messages; show them instead of a traceback.
        logger.error("%s", exc)
        return EXIT_PROBLEM


if __name__ == "__main__":
    raise SystemExit(main())
