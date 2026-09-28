"""Command-line entry point.

Phase 0 scope: argument parsing, logging setup, ``--version``, and ``--doctor``
(an offline environment self-check). Retrieval/LLM/voice modes arrive in later
phases — nothing here pretends to work before then.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence

from . import __version__
from .config import Settings
from .errors import BankVoiceAssistantError
from .logging_setup import configure_logging

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_USAGE = 2


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
        print(f"[PEND] knowledge base   not authored yet (Phase 1) — {settings.paths.kb_dir}")

    if (
        settings.paths.index_dir.is_dir()
        and (settings.paths.index_dir / "chunks.json").is_file()
        and (settings.paths.index_dir / "embeddings.npy").is_file()
        and (settings.paths.index_dir / "bm25.json").is_file()
    ):
        print(f"[OK  ] retrieval index  {settings.paths.index_dir}")
    else:
        print(f"[PEND] retrieval index  not built yet (Phase 2) — {settings.paths.index_dir}")

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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for both ``bank-voice`` and ``python -m bank_voice_assistant``."""
    args = build_parser().parse_args(argv)
    logger = configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        settings = Settings.from_env()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("effective settings: %s", settings.as_log_dict())

        if args.doctor:
            return run_doctor(settings)

        if args.build_index:
            return run_build_index(settings)

        print("Nothing to run yet — voice/ask modes arrive in later phases. Try --doctor or --build-index.")
        return EXIT_OK
    except BankVoiceAssistantError as exc:
        # Project errors carry actionable messages; show them instead of a traceback.
        logger.error("%s", exc)
        return EXIT_PROBLEM


if __name__ == "__main__":
    raise SystemExit(main())
