# 🎙️ Conversational Voice RAG Agent

An end-to-end voice-in, voice-out AI assistant with pluggable knowledge retrieval, designed for fast, grounded, and conversational voice interactions.

---

## 🌟 Key Features

- **Pluggable Knowledge Base**: Easily drop your custom Markdown documents into `data/kb/` and re-index in seconds for any domain (banking, healthcare, legal, customer support).
- **Voice-Optimized Responses**: Generates natural, concise spoken answers ($\le 35$ words / $\le 3$ sentences) specifically designed for audio playback.
- **Hybrid Retrieval (Dense + Sparse)**: Combines semantic vector embeddings (`fastembed` ONNX) with exact keyword matching (`BM25`) using score fusion and confidence gating.
- **Deterministic Guardrails**: Automated numeric and fact verification ensures generated answers are strictly grounded in retrieved reference context before speech synthesis.
- **Low-Latency Voice Loop**: End-to-end turnaround under 3 seconds on standard hardware with dynamic ambient-noise voice activity detection (VAD).

---

## 🛠️ Architecture Overview

```
[ User Microphone ]
        │  (16 kHz Mono Audio)
        ▼
1. Audio Transport & VAD
   • WebRTC VAD with adaptive RMS noise gating
   • 800 ms trailing silence endpoint detection
        │
        ▼
2. Speech-to-Text (STT)
   • Faster-Whisper with domain vocabulary biasing
        │
        ▼
3. Safety Check & Hybrid Retrieval
   • Pre-retrieval prompt injection / instruction-override defense
   • Dense Vector Search: FastEmbed (`BAAI/bge-small-en-v1.5` ONNX)
   • Sparse Lexical Search: Pure-Python BM25
   • 50/50 score fusion with minimum confidence gating (min_score: 0.45)
        │
        ▼
4. Grounded LLM Generation
   • Ollama (`llama3.2:latest`) with temperature pinned to 0.0
   • Strict context-grounding prompt (outputs REFUSE if facts are absent)
        │
        ▼
5. Numeric & Fact Guardrail Verifier
   • Verifies all numbers, percentages, and currencies match retrieved sources
   • Automatic corrective retry loop with safe refusal fallback
        │
        ▼
6. Text-to-Speech (TTS)
   • Piper TTS (`en_US-lessac-medium.onnx` @ 22,050 Hz)
        │
        ▼
[ Speaker Output ]
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running in the background:
  ```bash
  ollama pull llama3.2
  ```
- **Piper Voice Model**: Download `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json` from the [Piper Voices repository](https://github.com/rhasspy/piper/blob/master/VOICES.md) and place them in the project root directory.

### 2. Installation
Clone the repository and install dependencies in an editable virtual environment:

```bash
git clone https://github.com/Harshsinghrajpurohit/AI--Voice--Agent.git
cd AI--Voice--Agent

# Create and activate virtual environment
python -m venv test
.\test\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Verify Environment
Run the built-in diagnostic tool to verify audio devices, models, and dependencies:

```bash
bank-voice --doctor
```

---

## 💻 Usage

### Build / Rebuild Knowledge Base Index
Whenever you add or modify Markdown files in `data/kb/`, compile the hybrid index:

```bash
bank-voice --build-index
```

### One-Shot Text Query (`--ask`)
Test retrieval, generation, and guardrails without opening the microphone:

```bash
bank-voice --ask "What are the rules for opening an account?"
```

### Interactive Hands-Free Voice Mode (`--voice`)
Start the continuous voice loop (speak when prompted; say "exit", "quit", or "stop" to finish):

```bash
bank-voice --voice
```

Limit to a specific number of turns:
```bash
bank-voice --voice --turns 5
```

---

## 📂 Customizing the Knowledge Base

To adapt the assistant to your own domain or dataset:

1. Place your Markdown files in `data/kb/`.
2. Include YAML front-matter with document metadata:
   ```markdown
   ---
   title: Product FAQ
   category: Retail
   tags: [pricing, policy]
   last_updated: 2026-10-01
   ---

   ## Refund Policy
   Refunds are processed within 7 business days with a valid receipt.
   ```
3. Run `bank-voice --build-index` to regenerate the hybrid embeddings and BM25 index.

---

## 🧪 Testing & Evaluation

### Run Unit Tests
The test suite contains 170+ deterministic tests that execute in under 2 seconds:

```bash
pytest
```

### Run Golden Benchmark Suite
Evaluate answer accuracy, retrieval precision, refusal recall, and latency across the evaluation dataset:

```bash
bank-voice --eval --report run.json
```

---

## ⚙️ Configuration

Configuration is managed via typed settings in `src/bank_voice_assistant/config.py` and can be overridden using environment variables:

| Variable | Default | Description |
|---|---|---|
| `BVA_LLM_MODEL` | `llama3.2` | Ollama model identifier |
| `BVA_TOP_K` | `4` | Number of retrieved context chunks |
| `BVA_MIN_SCORE` | `0.45` | Minimum similarity threshold for retrieval |
| `BVA_HYBRID_ALPHA` | `0.5` | Weight between dense (1.0) and sparse (0.0) search |
| `BVA_LATENCY_TARGET_S` | `4.0` | Target turnaround latency (seconds) |
| `BVA_LATENCY_CEILING_S` | `8.0` | Maximum acceptable latency ceiling (seconds) |

---

## 📄 License
This project is open-source under the MIT License.