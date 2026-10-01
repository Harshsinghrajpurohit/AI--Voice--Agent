# Memory.md — project context for any AI tool or new chat

> Update at the end of every work session. Exists so a fresh chat (or a different AI tool)
> can continue without re-reading the whole codebase or inventing things.
> Last updated: 2026-10-01 — end of Phase 7.

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
- VAD: webrtcvad aggressiveness 3 + an RMS gate that self-calibrates to the live microphone's
  ambient noise at the start of every turn (floor 120, noise x2.0, ceiling 800), so one setting
  works with the internal array or a headset, headphones or not; set `BVA_CALIBRATION_MS=0` to pin
  the old fixed 800 gate. 800 ms trailing silence ends the turn; 15 s recording cap; 5 frames to arm.
- Latency budget: <= 4 s target, <= 8 s hard ceiling. It now lives in `Settings.pipeline`
  (`BVA_LATENCY_TARGET_S` / `BVA_LATENCY_CEILING_S`) and every turn logs a per-stage breakdown
  (VAD + STT + retrieval + LLM + TTS); **playback is excluded** from the budget on purpose.
- Orchestration: `bank_voice_assistant.pipeline.VoicePipeline` is the only place the stages are
  wired together. All collaborators are injectable and built lazily, so importing it (or the CLI)
  never loads a model, opens the mic, or needs the index.
- Voice loop: `bank-voice --voice` (alias `--listen`, optional `--turns N`) runs until an exit
  phrase (`exit` / `quit` / `stop` / `goodbye`, overridable with `BVA_EXIT_PHRASES`) or Ctrl+C.
  `--ask "..."` is the one-shot text turn. Silent or failing turns are retried up to
  `silent_turn_retries` times in a row, so an unattended session can never spin forever.
- Console output is written with `errors="replace"` (`cli.configure_console`): KB answers contain
  `₹`, which a Windows cp1252 console cannot encode.
- Version: `pyproject.toml` and `__init__.__version__` must match (enforced by a test). Now **0.4.0**.

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
- Phase 4 (Numeric & Verbatim Verification Guardrails): DONE — verbatim number and phone sequence extractor,
  `NumericVerifier` with dual string & float normalization matching, `GuardedGenerator` orchestrating strict corrective
  retry and safe refusal fallback, `bank-voice --query <text>` CLI option, 38 deterministic tests passing.
- Phase 5 (Voice Pipeline — STT, TTS, Audio Transport): DONE — `Transcriber` wrapping Faster-Whisper with banking domain
  prompt biasing, `PiperTTS` subprocess wrapper with length scaling and validation, `AudioTransport` encapsulating
  `sounddevice` mic capture with WebRTC VAD endpointing and RMS energy gating, 47 deterministic tests passing.
- Phase 6 (End-to-End Orchestration & Interactive Loop): DONE — `pipeline/` package with
  `TurnTelemetry` / `TurnResult` / `VoicePipeline` (lazy, injectable stages), `bank-voice --voice`
  / `--listen` / `--turns N` interactive loop with graceful per-turn error recovery,
  `--ask` re-routed through the pipeline, per-stage latency telemetry + budget warnings,
  `--doctor` now also verifies audio devices and prints the latency budget, 76 deterministic tests
  passing in ~1.0 s. Live-verified on this machine (no cloud): Piper question -> Whisper
  round-trip verbatim -> retrieval 0.46 s -> grounded INR answer -> Piper WAV; warm turn ≈ 2.9 s.
- Phase 7 (Evaluation, Benchmarking & Hardening): DONE — golden dataset (61 rows: 39 answerable /
  22 must_refuse across 8 categories) with a validated JSONL loader, pure-function scoring metrics
  (`eval/metrics.py`), pass/fail floors and ceilings in `EvalSettings`, and the replay harness
  (`eval/runner.py`; `bank-voice --eval [--report PATH]`, exit code 3 on a missed threshold). Step 7.6
  added the instruction-override defence — `guardrails.detect_instruction_override` refuses overrides
  *before* retrieval and `announces_rule_change` catches figure-free compliance lines in
  `GuardedGenerator` — which retired the `adv-dev-mode-01` known failure and raised
  `min_refusal_recall` to 1.0. 178 deterministic tests passing; the live 61-row acceptance run still
  needs a built index + a running Ollama server.

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
- `setuptools` was installed into the venv (Python 3.13 venvs don't ship it) so
  `python -m pip install -e . --no-deps --no-build-isolation` works offline; re-run that after any
  `pyproject.toml` version change or `importlib.metadata.version()` stays stale.
- `.cache\index` is built (chunks/embeddings/BM25), so `--ask` works end-to-end. Rebuild with
  `bank-voice --build-index` after editing `data/kb`. Both `.cache/` and the index are gitignored.
- Live timings (i5-11400H, CPU int8 Whisper, Vulkan Ollama): cold Ollama load ~13 s; warm turn
  STT 0.82 s + retrieval 0.46 s + LLM 1.36 s + Piper ≈ 0.3 s ≈ 2.9 s. The very first
  `--ask` after an Ollama restart can breach the 8 s ceiling — the telemetry warning says so.

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
- A Windows cp1252 console cannot encode `₹`: printing a valid KB answer used to raise
  `UnicodeEncodeError` and abort the turn. `cli.configure_console()` sets `errors="replace"` on
  stdout/stderr — lossy, never fatal. Anything printed by the CLI goes through it.
- Modules log via `logging.getLogger(__name__)`, i.e. `bank_voice_assistant.*` children. Configuring
  only the short `bva` logger silently dropped every INFO/DEBUG line (including turn telemetry);
  `configure_logging()` now configures the package logger too.
- Chinese/`₹`-style characters are safe for Piper (stdin is UTF-8) but not for the console — a
  lossy console is expected, the spoken answer is not affected.
- pytest runs in file order; a logging handler left pointing at a closed capture stream can make
  later tests noisy, so `tests/test_pipeline.py` clears the handler via an autouse fixture.
- Inserting into a file with the editor tool needs the *actual* line count (`Get-Content | Measure`)
  or the content lands mid-function — verify with `Get-Content | Select-Object -Skip N` afterwards.
- A single fixed RMS gate cannot fit both mics at once: on the internal array most speech frames sit
  below the old 800 gate, so voice mode kept answering "I did not catch that". The energy gate now
  calibrates to ambient noise per turn (`transport/local_audio.py`); verify a quiet mic with
  `python -c "import sounddevice as sd, numpy as np; a=sd.rec(int(5*16000),samplerate=16000,channels=1,dtype='int16'); sd.wait(); print('rms', round(float(np.sqrt(np.mean(a.astype(float)**2))),1))"`.
- faster-whisper assumes any numpy array it is handed is already float32 in [-1, 1]; `record_turn`
  yields int16, so passing the array straight through transcribes to garbage or to nothing — the
  "Speech detected but nothing was transcribed" symptom. `Transcriber` scales it at the boundary
  (`stt.to_float_waveform`). A file path still works untouched.
