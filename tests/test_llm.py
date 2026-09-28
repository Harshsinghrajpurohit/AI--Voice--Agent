"""Tests for grounded prompt builder and Ollama generator wrapper."""

from __future__ import annotations

import pytest

from bank_voice_assistant.config import Settings
from bank_voice_assistant.errors import LLMError
from bank_voice_assistant.kb import Chunk
from bank_voice_assistant.llm import (
    REFUSAL_TOKEN,
    STANDARD_REFUSAL,
    GroundedGenerator,
    clean_spoken_text,
)
from bank_voice_assistant.retrieval import RetrievedChunk


class FakeOllamaClient:
    """Mock Ollama client for deterministic testing."""

    def __init__(self, response_text: str = "Personal loan interest rates start at 10.50 percent.") -> None:
        self.response_text = response_text
        self.last_messages: list[dict[str, str]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs) -> dict:
        self.last_messages = messages
        return {"message": {"content": self.response_text}}


def test_clean_spoken_text() -> None:
    raw = "**Northwind Classic** savings has a *minimum* balance of `5,000` rupees."
    cleaned = clean_spoken_text(raw)
    assert cleaned == "Northwind Classic savings has a minimum balance of 5,000 rupees."
    assert "*" not in cleaned
    assert "`" not in cleaned


def test_prompt_builder_contains_chunks() -> None:
    settings = Settings()
    gen = GroundedGenerator(settings)
    chunk = Chunk(
        chunk_id="chk1",
        doc_title="Loans Guide",
        category="Loans",
        heading="Auto Loans",
        content="Auto loan rates start at 8.75% for up to 7 years.",
    )
    rc = RetrievedChunk(chunk=chunk, score=0.88, dense_score=0.9, sparse_score=0.8)
    messages = gen.build_prompt("What is the auto loan rate?", [rc])

    assert len(messages) == 2
    system_msg = messages[0]["content"]
    assert "Loans Guide" in system_msg
    assert "Auto Loans" in system_msg
    assert "8.75%" in system_msg
    assert "35 words" in system_msg
    assert messages[1]["content"] == "What is the auto loan rate?"


def test_generate_empty_chunks_refuses_immediately() -> None:
    settings = Settings()
    client = FakeOllamaClient()
    gen = GroundedGenerator(settings, client=client)

    result = gen.generate("What is the rate?", [])
    assert result.is_refusal is True
    assert result.text == STANDARD_REFUSAL
    assert len(client.last_messages) == 0


def test_generate_handles_refusal_token() -> None:
    settings = Settings()
    client = FakeOllamaClient(response_text=REFUSAL_TOKEN)
    gen = GroundedGenerator(settings, client=client)
    chunk = Chunk("c1", "Title", "Cat", "H", "Some unrelated banking text.")
    rc = RetrievedChunk(chunk=chunk, score=0.7, dense_score=0.7, sparse_score=0.7)

    result = gen.generate("What is the weather today?", [rc])
    assert result.is_refusal is True
    assert result.text == STANDARD_REFUSAL


def test_generate_success() -> None:
    settings = Settings()
    expected = "Personal loan interest rates start at 10.50 percent."
    client = FakeOllamaClient(response_text=f"**{expected}**")
    gen = GroundedGenerator(settings, client=client)
    chunk = Chunk("c1", "Title", "Cat", "H", "Personal loans start at 10.50%.")
    rc = RetrievedChunk(chunk=chunk, score=0.85, dense_score=0.85, sparse_score=0.85)

    result = gen.generate("What is the personal loan rate?", [rc])
    assert result.is_refusal is False
    assert result.text == expected
    assert result.word_count == len(expected.split())


def test_generate_raises_llm_error_on_client_exception() -> None:
    settings = Settings()

    class ErrorClient:
        def chat(self, **kwargs):
            raise ConnectionError("Ollama daemon unreachable")

    gen = GroundedGenerator(settings, client=ErrorClient())
    chunk = Chunk("c1", "Title", "Cat", "H", "Some text.")
    rc = RetrievedChunk(chunk=chunk, score=0.8, dense_score=0.8, sparse_score=0.8)

    with pytest.raises(LLMError) as exc_info:
        gen.generate("What is the interest rate?", [rc])
    assert "Ollama generation failed" in str(exc_info.value)
