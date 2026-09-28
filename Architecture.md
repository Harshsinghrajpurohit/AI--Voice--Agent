# Architecture.md — System Architecture & Flow

## 1. High-Level Architecture Flow

```
+---------------------------------------------------------------------------------+
|                                 USER (SPEECH)                                   |
+---------------------------------------------------------------------------------+
                                      | Audio Stream (16 kHz, Mono, 16-bit PCM)
                                      v
+---------------------------------------------------------------------------------+
| Audio Transport & Endpointing (`bank_voice_assistant.transport.local_audio`)     |
|   - webrtcvad (Aggressiveness 3)                                                |
|   - RMS energy gate (800)                                                       |
|   - 800ms trailing silence trigger                                              |
+---------------------------------------------------------------------------------+
                                      | Complete Utterance Buffer
                                      v
+---------------------------------------------------------------------------------+
| STT Engine (`bank_voice_assistant.stt`)                                         |
|   - faster-whisper `base` (int8 on CPU)                                         |
|   - Banking prompt prefix for domain term biasing                               |
+---------------------------------------------------------------------------------+
                                      | Transcribed Text
                                      v
+---------------------------------------------------------------------------------+
| Hybrid Retrieval Engine (`bank_voice_assistant.retrieval`)                      |
|   - Dense Embeddings: fastembed (`BAAI/bge-small-en-v1.5`)                      |
|   - Sparse / Lexical: BM25 index                                                |
|   - Hybrid scoring (Alpha 0.5) & Min Score Threshold Check                      |
|   - KB Store: `data/kb/*.md` -> Pre-built index in `.cache/index`               |
+---------------------------------------------------------------------------------+
         | Score >= threshold                  | Score < threshold
         v                                     v
+---------------------------------------+  +--------------------------------------+
| LLM Generator                         |  | Fallback Handler                     |
| (`bank_voice_assistant.llm`)          |  |   - Return standard refusal phrase:  |
|   - Ollama `llama3.2:latest`          |  |     "I apologize, but I do not have  |
|   - Vulkan acceleration               |  |      information regarding that in   |
|   - Strict system grounding prompt    |  |      my banking records."            |
|   - Temp: 0.0, <= 35 words            |  +--------------------------------------+
+---------------------------------------+                      |
         | Raw Answer Text                                     |
         v                                                     |
+--------------------------------------------------------+     |
| Grounding Guardrail & Numeric Verifier                 |     |
| (`bank_voice_assistant.guardrails`)                    |     |
|   - Extract all numbers/rates/dates                    |     |
|   - Exact match check against retrieved context chunks |     |
|   - One strict regeneration attempt if ungrounded      |     |
+--------------------------------------------------------+     |
         | Validated Spoken Text                               |
         +--------------------------+--------------------------+
                                    |
                                    v

## 2. Directory & Module Structure

```
d:\Work Folder\Voice Agents\
├── .cache/                     # Local transient caches (pip, index embeddings)
├── data/
│   ├── kb/                     # Curated markdown FAQ documents with front-matter
│   └── eval/                   # Golden benchmark queries and ground-truth pairs
├── src/
│   └── bank_voice_assistant/
│       ├── __init__.py         # Package exports & version
│       ├── __main__.py         # CLI executable entry point (`python -m bank_voice_assistant`)
│       ├── cli.py              # CLI argument parser and subcommands
│       ├── config.py           # Typed immutable settings from defaults/env
│       ├── errors.py           # Unified exception hierarchy
│       ├── logging_setup.py    # Structured logging configuration
│       ├── kb/                 # Phase 1: KB schema, document parser, validator
│       ├── retrieval/          # Phase 2: Vector embedding, BM25, hybrid indexer
│       ├── llm/                # Phase 3: Ollama client, grounded prompt builder
│       ├── guardrails/         # Phase 4: Numeric verifier and refusal logic
│       ├── tts/                # Phase 5: Piper subprocess integration
│       ├── stt/                # Phase 5: Faster-whisper loader & transcriber
│       ├── transport/          # Phase 5: Sounddevice mic capture & VAD stream
│       ├── pipeline/           # Phase 6: End-to-end voice loop orchestrator
│       └── eval/               # Phase 7: Evaluation runner & latency measurement
├── tests/                      # Pytest suite mirror
├── pyproject.toml              # Build definition and dependencies
├── PRD.md                      # Product requirements document
├── Architecture.md             # System architecture & component interfaces
├── Rules.md                    # Engineering boundaries & constraints
├── Phases.md                   # Phased implementation roadmap
├── Design.md                   # Conversational UX, prompt & speech design
└── Memory.md                   # Continuous session progress tracker
```

## 3. Core Component Contracts & Interfaces

### 3.1 Knowledge Base Document (`bank_voice_assistant.kb`)
- Input: `.md` files containing strict key-value front-matter (`title`, `category`, `tags`, `last_updated`).
- Chunker: Split by Markdown H2/H3 sections, retaining front-matter context in every chunk metadata.

### 3.2 Index & Retriever (`bank_voice_assistant.retrieval`)
- Interface: `search(query: str, top_k: int = 4) -> list[RetrievedChunk]`
- Returns ranked chunks with dense similarity scores, BM25 scores, and fused hybrid scores.

### 3.3 LLM Grounding & Generator (`bank_voice_assistant.llm`)
- Prompting: Strictly injects retrieved chunks into system context.
- System prompt instructs: *Answer in <= 35 words. Speak naturally for audio. Rely solely on provided chunks. If not found, return EXACT refusal token.*

### 3.4 Guardrail Verifier (`bank_voice_assistant.guardrails`)
- Regex extraction of numeric literals: percentages (`5.5%`), currency (`$500`), counts, and dates.
- Assertion: `all(num in context_text for num in generated_nums)`.

+---------------------------------------------------------------------------------+
| Speech Synthesis Engine (`bank_voice_assistant.tts`)                            |
|   - Piper TTS (`en_US-lessac-medium.onnx`)                                      |
|   - 22,050 Hz playback stream                                                   |
+---------------------------------------------------------------------------------+
                                    | Output Audio Waveform
                                    v
+---------------------------------------------------------------------------------+
| Speaker Output (`sounddevice`)                                                  |
+---------------------------------------------------------------------------------+
```
