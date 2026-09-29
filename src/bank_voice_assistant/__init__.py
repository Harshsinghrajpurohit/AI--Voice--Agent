"""
Bank Voice Assistant — a fully local, knowledge-base-grounded voice agent.

Design goal: answer banking FAQ questions (rates, fees, policies) using ONLY the
curated knowledge base in ``data/kb``. Anything the KB does not cover is refused
rather than invented.

Pipeline: mic -> WebRTC VAD endpointing -> Faster-Whisper (STT) -> hybrid retrieval
(local sentence embeddings + BM25) -> Ollama/Llama 3.2 with a grounded prompt ->
numeric grounding verification -> Piper (TTS) -> speaker.
"""

__version__ = "0.4.0"

__all__ = ["__version__"]
