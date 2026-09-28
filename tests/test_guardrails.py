"""Tests for numeric grounding verification and guardrail retry logic."""

from __future__ import annotations

import pytest

from bank_voice_assistant.config import Settings
from bank_voice_assistant.errors import GuardrailViolation
from bank_voice_assistant.guardrails import (
    GuardedGenerator,
    NumericVerifier,
    extract_numbers,
)
from bank_voice_assistant.kb import Chunk
from bank_voice_assistant.llm import STANDARD_REFUSAL, GroundedGenerator
from bank_voice_assistant.retrieval import RetrievedChunk


def test_extract_numbers() -> None:
    text = "Call 1800-419-0022. Home loan rate is 8.40% for ₹50,000 to ₹5,00,000."
    nums = extract_numbers(text)
    assert "1800-419-0022" in nums
    assert "8.40" in nums
    assert "50,000" in nums
    assert "5,00,000" in nums


def test_verifier_grounded_numbers() -> None:
    verifier = NumericVerifier()
    context = "The Classic Savings Account requires a minimum balance of ₹5,000 with 3.50% interest."

    # Grounded response with variations
    response = "The minimum balance is 5,000 rupees and interest is 3.50 percent."
    res = verifier.verify(response, context)
    assert res.is_valid is True
    assert len(res.ungrounded_numbers) == 0


def test_verifier_catches_hallucinated_number() -> None:
    verifier = NumericVerifier()
    context = "Home loan interest rate is 8.40% per annum."

    # Hallucinated rate 9.20%
    response = "The home loan interest rate is 9.20%."
    res = verifier.verify(response, context)
    assert res.is_valid is False
    assert "9.20" in res.ungrounded_numbers


def test_check_or_raise_raises_guardrail_violation() -> None:
    verifier = NumericVerifier()
    context = "Fixed deposit tenure is 12 months."
    with pytest.raises(GuardrailViolation):
        verifier.check_or_raise("Tenure is 24 months.", context)


class SequentialFakeClient:
    """Mock client returning sequential responses for testing retries."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, **kwargs) -> dict:
        reply = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return {"message": {"content": reply}}


def test_guarded_generator_retry_success() -> None:
    settings = Settings()
    client = SequentialFakeClient([
        "The loan rate is 9.00 percent.",
        "The loan rate is 8.40 percent.",
    ])
    raw_gen = GroundedGenerator(settings, client=client)
    guarded_gen = GuardedGenerator(settings, raw_gen)

    chunk = Chunk("c1", "Loans", "Loans", "Home Loan", "Home loan interest rate is 8.40%.")
    rc = RetrievedChunk(chunk=chunk, score=0.9, dense_score=0.9, sparse_score=0.9)

    res = guarded_gen.generate("What is the home loan rate?", [rc])
    assert res.is_refusal is False
    assert "8.40" in res.text
    assert client.call_count == 2


def test_guarded_generator_retry_fails_and_refuses() -> None:
    settings = Settings()
    client = SequentialFakeClient([
        "The loan rate is 9.00 percent.",
        "Actually, the loan rate is 9.50 percent.",
    ])
    raw_gen = GroundedGenerator(settings, client=client)
    guarded_gen = GuardedGenerator(settings, raw_gen)

    chunk = Chunk("c1", "Loans", "Loans", "Home Loan", "Home loan interest rate is 8.40%.")
    rc = RetrievedChunk(chunk=chunk, score=0.9, dense_score=0.9, sparse_score=0.9)

    res = guarded_gen.generate("What is the home loan rate?", [rc])
    assert res.is_refusal is True
    assert res.text == STANDARD_REFUSAL
    assert client.call_count == 2
