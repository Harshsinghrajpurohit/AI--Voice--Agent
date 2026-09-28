# Memory.md — project context for any AI tool or new chat

> Update at the end of every work session. Exists so a fresh chat (or a different AI tool)
> can continue without re-reading the whole codebase or inventing things.
> Last updated: 2026-09-28 — end of Phase 3.

## What this project is
Fully local, knowledge-base-grounded **banking FAQ** voice assistant. Answers only from
`data/kb`; refuses anything the KB does not support. No cloud, no vendor SDKs, no account data.

Pipeline: mic -> WebRTC VAD endpointing -> Faster-Whisper STT -> hybrid retrieval
(dense + BM25) -> guardrailed Ollama/Llama 3.2 -> numeric verification -> Piper TTS -> speakers.

## Locked decisions
- Bank identity: **Northwind Bank** (fictional retail & commercial bank).
- Market & Currency: **Indian Rupee (INR, ₹)** across all products, tariffs, and interest rates.
- Channel (v1): local device only. The audio transport is *pluggable*; a web call is a later
  phase. No SIP/PSTN/outbound — telephony can never be fully local (it needs a carrier).
- STT: faster-whisper `base`, CPU, int8, language `en`, banking-vocabulary `initial_prompt`.
- LLM: Ollama `llama3.2` on the **Vulkan** backend; temperature hard-pinned to 0.0 in config.
- Embeddings: fastembed (local ONNX) `BAAI/bge-small-en-v1.5`; retrieval is hybrid dense + BM25.
- Retrieval defaults: top_k=4, min_score=0.45, hybrid_alpha=0.5 — to be re-tuned in the eval phase.
- Grounding: refuse when retrieval is empty or borderline; every number must appear **verbatim**
  in a retrieved chunk; one stricter retry, then refuse (`GuardrailViolation`).
- Answers: <= 35 words / <= 3 sentences spoken, plus a fixed refusal sentence.
- TTS: Piper `en_US-lessac-medium` (22050 Hz). The model is a downloaded asset, NOT in git.
- VAD: webrtcvad aggressiveness 3 + RMS energy gate 800; 800 ms trailing silence ends the turn;
  15 s recording cap; 5 speech frames to arm.
- Latency budget: <= 4 s target, <= 8 s hard ceiling.
- Version: `pyproject.toml` and `__init__.__version__` must match (enforced by a test).

## Phase status
- Phase 0 (installable & testable): DONE — `config.py`, `cli.py`, `__main__.py`, `tests/`,
  `.gitignore`, `Memory.md`, `PRD.md`, `Architecture.md`, `Rules.md`, `Phases.md`, `Design.md`.
- Phase 1 (Knowledge Base & Ingestion): DONE — strict front-matter parser, markdown section chunker,
  6 curated Northwind Bank FAQ documents (INR) in `data/kb/`, 20 tests passing, `bank-voice --doctor` reports KB OK.
- Phase 2 (Hybrid Retrieval Indexer & Search Engine): DONE — pure-Python BM25 engine preserving decimals,
  dense FastEmbed (`bge-small-en-v1.5`) wrapper with normalized cosine scoring, linear score fusion with
  `min_score` thresholding, persistence to `.cache/index/`, `bank-voice --build-index` wired into CLI, 25 tests passing.
- Phase 3 (Grounded LLM Generation): DONE — Ollama client wrapper with `llama3.2`, strict grounding prompt
  with voice conciseness rules (<= 35 words / <= 3 sentences), markdown stripping for clean TTS output,
  empty-retrieval and token-based refusal handling (`STANDARD_REFUSAL`), 31 tests passing.
- Phases 4-7: pending. Next is Phase 4 (Numeric & Verbatim Verification Guardrails).

## Environment (this machine)
- Python 3.13.0. The venv folder is `test\` (NOT `tests\`); editable install points at `src`.
- i5-11400H (6C/12T) - 7.7 GB RAM - RTX 2050 4 GB VRAM - driver 592.00.
- Ollama 0.34.4 with user env var `OLLAMA_VULKAN=1` — its CUDA build crashes on model load
  (exit 0xc0000409 / "shared object initialization failed"); Vulkan gives ~22.6 tok/s vs ~4.5 on CPU.
  Restart the Ollama app after changing that variable.
- `llama3.2:latest` (2.0 GB) pulled; Whisper `base` cached in `~/.cache/huggingface`.
- `webrtcvad-wheels==2.0.14` provides the `webrtcvad` module.
- Piper voice files live in the repo root; `bank-voice.exe` is registered in the venv Scripts.
- pytest 9.1.1 and fastembed installed; `PIP_CACHE_DIR` points at the repo's `.cache/pip`.

## Conventions
- `Settings()` = code defaults; `Settings.from_env()` = defaults + `BVA_*` overrides. Nothing
  else reads the environment.
- Every failure raises a type from `errors.py`; the CLI prints the message, never a traceback.
- Only the audio-transport module may import `sounddevice`.
- Tests never touch the mic, Ollama, or model weights; use `monkeypatch` for env vars, never `$env:`.
- New dependencies require an explicit decision (memory budget: 7.7 GB).

## Gotchas learned the hard way
- A stale `BVA_*` variable left in the terminal silently changes behaviour — clean up after manual checks.
- VS Code creates *nested* folders when a new filename contains a path — verify with `Get-ChildItem -Name`.
- `fastembed` downloads its ONNX model on first use, not at install time.
- `ollama.chat` on a model that is not pulled returns HTTP 404 (`ResponseError`), not a connection error.
