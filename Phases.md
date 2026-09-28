# Phases.md — Implementation Roadmap

This project progresses through 8 structured phases (Phase 0 to Phase 7).

---

## Phase 0: Scaffolding, CLI, Environment & Testing Harness [COMPLETED]
- [x] Configure pyproject.toml, package metadata, and editable installation.
- [x] Implement robust typed immutable configuration (`Settings`) with environment variable overrides.
- [x] Implement exception hierarchy (`BankVoiceAssistantError`).
- [x] Implement structured logging with `-v` (verbose) and `-q` (quiet) CLI flags.
- [x] Implement `--doctor` environment self-check command.
- [x] Write baseline test suite for configuration and CLI argument parsing.
- [x] Establish repository hygiene and `.gitignore` stripping binaries and caches.

---

## Phase 1: Knowledge Base Content & Ingestion
- [ ] Define markdown front-matter schema (`title`, `category`, `tags`, `last_updated`).
- [ ] Implement robust front-matter parser and markdown document chunker (H2/H3 level).
- [ ] Author comprehensive authoritative FAQ documents across core banking domains:
  - Savings & Current Accounts
  - Fixed Deposits & Recurring Deposits
  - Credit Cards, Fees, & Rewards
  - Loans (Personal, Auto, Home) & Interest Rates
  - Account Opening, KYC & Verification
  - Digital Banking, ATM/Debit Card Blocking & Emergency Fraud Contacts
- [ ] Write schema validation and chunk integrity unit tests.

---

## Phase 2: Hybrid Retrieval Engine
- [ ] Implement dense embedding generation via `fastembed` (`BAAI/bge-small-en-v1.5`).
- [ ] Implement sparse BM25 lexical index over chunk tokens.
- [ ] Build hybrid reciprocal rank fusion (RRF) / weighted score merger.
- [ ] Implement index persistence and loading under `.cache/index/`.
- [ ] Expose `bank-voice --build-index` CLI command.
- [ ] Add unit tests verifying hybrid retrieval precision and score filtering.

---

## Phase 3: Grounded LLM Generation
- [ ] Integrate local Ollama client (`llama3.2`) with Vulkan backend support.
- [ ] Design rigid grounding system prompt enforcing the 35-word / 3-sentence voice constraint.
- [ ] Implement prompt chunk-context injection with citation tracking.
- [ ] Handle unanswerable queries by emitting standardized refusal tokens.
- [ ] Unit tests for prompt building and Ollama response parsing (mocked client).

---

## Phase 4: Verification & Guardrails
- [ ] Build regex-based numeric and rate verifier (`verify_grounding(response, context)`).
- [ ] Implement automatic one-time strict retry logic when hallucination or ungrounded number is detected.
- [ ] Fall back to safe refusal upon repeated guardrail failure.
- [ ] Comprehensive unit tests for strict number verification and edge cases.

---

## Phase 5: Voice Pipeline (STT, TTS, Audio Transport)
- [ ] Wrap `faster-whisper` STT with domain prompt biasing.
- [ ] Integrate `piper-tts` subprocess streaming / synthesis with `en_US-lessac-medium`.
- [ ] Refactor VAD capture into clean, testable `transport/local_audio.py` module.
- [ ] Unit tests for STT/TTS adapters with mocked binaries.

---

## Phase 6: End-to-End Orchestration & Interactive Loop
- [ ] Combine all stages into a unified `VoicePipeline` orchestrator.
- [ ] Add CLI commands: `bank-voice --voice` (interactive audio loop) and `bank-voice --ask "question"` (one-shot text query).
- [ ] Implement turn latency telemetry logging breakdown (VAD + STT + Retrieval + LLM + TTS).

---

## Phase 7: Evaluation, Benchmarking & Hardening
- [ ] Golden dataset with 50+ banking questions (factual, out-of-scope, adversarial).
- [ ] Automated evaluation script measuring Grounding Precision, Refusal Accuracy, and Latency breakdown.
- [ ] Final performance tuning to ensure < 4.0s average response time on reference hardware.
