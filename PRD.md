# PRD — Product Requirements Document: Bank Voice Assistant

## 1. Executive Summary & Objective
A fully local, zero-cloud, knowledge-base-grounded voice agent built specifically for banking FAQ inquiries (accounts, cards, loans, fees, procedures). The system transcribes spoken user queries, retrieves authoritative policy answers from a local curated knowledge base, generates strictly grounded responses via a local LLM, and speaks the answer back via a local TTS engine.

**Core Tenet:** 100% privacy, zero recurring vendor API costs, zero data leakage, and rigorous anti-hallucination guardrails (refuse rather than guess or hallucinate).

---

## 2. Target Users & Personas
- **Bank Customers**: Inquiring about interest rates, balance minimums, card replacement fees, loan terms, and branch/operating policies over a natural voice interface.
- **Privacy-Sensitive Deployments**: On-premise branch kiosks, edge teller assists, or secure internal environments where audio and banking queries cannot touch external cloud networks.

---

## 3. Scope & Key Features

### 3.1 In-Scope (Phase 0–7)
- **Local Voice Pipeline**:
  - WebRTC Voice Activity Detection (VAD) with RMS energy gating.
  - Local speech-to-text (STT) via `faster-whisper`.
  - Local text-to-speech (TTS) via `piper-tts` using `en_US-lessac-medium`.
- **Authoritative Banking Knowledge Base (KB)**:
  - Curated markdown documents with strict metadata front-matter covering standard retail banking products:
    - Savings & Current Accounts
    - Fixed Deposits (CDs)
    - Credit Cards & Reward Programs
    - Loans (Personal, Auto, Home)
    - Fees, Tariffs, & Minimum Balance Penalties
    - Account Opening & KYC Requirements
    - Digital Banking, Security & Card Blocking
- **Hybrid Retrieval Engine**:
  - Dense vector retrieval (`fastembed` with `bge-small-en-v1.5` ONNX model).
  - Sparse lexical retrieval (BM25) with Reciprocal Rank Fusion (RRF) or linear hybrid weighting.
  - Strict score thresholds to detect out-of-scope or unanswerable queries.
- **Grounded Response Generation**:
  - Local LLM via Ollama (`llama3.2:latest`) locked to temperature `0.0`.
  - Concise spoken output formatting (<= 35 words / <= 3 sentences).
  - Numeric verification guardrails: all numbers, percentages, currency amounts, and dates in the output must match retrieved source chunks verbatim.
  - Safe fallbacks: standard refusal phrase when evidence is missing or ambiguous.
- **Verification & Evaluation**:
  - Synthetic golden question-answer test sets.
  - Automated evaluation harness checking retrieval precision, response latency, and numeric grounding accuracy.

### 3.2 Out-of-Scope (Non-Goals for v1)
- Live account transactions (e.g., money transfer, balance lookup requiring core-banking API integration).
- Multi-party telephony / SIP / PSTN trunks (v1 target is local workstation audio transport; web/telephony endpoints are pluggable future transports).
- Cloud AI APIs (OpenAI, Anthropic, ElevenLabs, Deepgram).

---

## 4. Non-Functional Requirements
- **Target Spoken Latency**: <= 4.0s from speech stop to first audio playback; hard ceiling <= 8.0s on reference hardware (RTX 2050 4GB + i5-11400H).
- **Grounding Accuracy**: 0% hallucinated rates, zero tolerated invented numeric values.
- **Resilience**: Graceful recovery from audio device errors, missing index files, or local Ollama disconnection.
