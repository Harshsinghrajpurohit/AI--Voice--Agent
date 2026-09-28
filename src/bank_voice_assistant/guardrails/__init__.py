"""Numeric grounding verification and guardrail enforcement."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from ..config import Settings
from ..errors import GuardrailViolation
from ..llm import STANDARD_REFUSAL, GenerationResult, GroundedGenerator
from ..retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

PHONE_PATTERN = re.compile(r"\b\d{3,4}[-\s]\d{3,4}[-\s]\d{4}\b")
NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")


def extract_numbers(text: str) -> list[str]:
    """Extract all numeric representations and phone sequences from text."""
    found: list[str] = []

    for m in PHONE_PATTERN.finditer(text):
        found.append(m.group(0))

    masked = PHONE_PATTERN.sub(" ", text)

    for m in NUMBER_PATTERN.finditer(masked):
        found.append(m.group(0))

    return found


def _to_float(raw: str) -> float | None:
    """Parse a number string stripped of commas into float, or None if invalid."""
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of numeric grounding verification."""

    is_valid: bool
    response_numbers: tuple[str, ...]
    context_numbers: tuple[str, ...]
    ungrounded_numbers: tuple[str, ...]

class NumericVerifier:
    """Verifies that all numbers in an LLM response are strictly present in source context."""

    def verify(self, response_text: str, context: str | Sequence[RetrievedChunk]) -> VerificationResult:
        """Check all numbers in response_text against the provided context."""
        if isinstance(context, str):
            context_text = context
        else:
            context_text = "\n".join(rc.chunk.content for rc in context)

        resp_nums = extract_numbers(response_text)
        if not resp_nums:
            return VerificationResult(
                is_valid=True,
                response_numbers=(),
                context_numbers=tuple(extract_numbers(context_text)),
                ungrounded_numbers=(),
            )

        ctx_nums = extract_numbers(context_text)
        ctx_raw_set = {n.lower() for n in ctx_nums}
        ctx_clean_set = {n.replace(",", "").replace("-", "").replace(" ", "").lower() for n in ctx_nums}
        ctx_floats = {val for n in ctx_nums if (val := _to_float(n)) is not None}

        ungrounded: list[str] = []

        for r_num in resp_nums:
            r_lower = r_num.lower()
            r_clean = r_lower.replace(",", "").replace("-", "").replace(" ", "")
            r_float = _to_float(r_num)

            if r_lower in ctx_raw_set or r_clean in ctx_clean_set:
                continue

            if r_float is not None and r_float in ctx_floats:
                continue

            escaped = re.escape(r_num)
            if re.search(r"(?<!\d)" + escaped + r"(?!\d)", context_text):
                continue

            ungrounded.append(r_num)

        return VerificationResult(
            is_valid=len(ungrounded) == 0,
            response_numbers=tuple(resp_nums),
            context_numbers=tuple(ctx_nums),
            ungrounded_numbers=tuple(ungrounded),
        )

    def check_or_raise(self, response_text: str, context: str | Sequence[RetrievedChunk]) -> None:
        """Verify grounding and raise GuardrailViolation if ungrounded numbers exist."""
        result = self.verify(response_text, context)
        if not result.is_valid:
            raise GuardrailViolation(f"Ungrounded numbers detected: {list(result.ungrounded_numbers)}")


class GuardedGenerator:
    """Orchestrates generation with numeric verification and automatic retry."""

    def __init__(
        self,
        settings: Settings,
        generator: GroundedGenerator,
        verifier: NumericVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.verifier = verifier or NumericVerifier()

    def generate(self, query: str, chunks: Sequence[RetrievedChunk]) -> GenerationResult:
        """Generate response with verification and safe fallback on violation."""
        if not chunks:
            return GenerationResult(
                text=STANDARD_REFUSAL,
                is_refusal=True,
                word_count=len(STANDARD_REFUSAL.split()),
                context_chunks=(),
            )

        # 1. Primary generation attempt
        result = self.generator.generate(query, chunks)
        if result.is_refusal:
            return result

        # 2. Verify numeric grounding
        verification = self.verifier.verify(result.text, chunks)
        if verification.is_valid:
            return result

        logger.warning(
            "Guardrail violation on first attempt: ungrounded numbers %s",
            verification.ungrounded_numbers,
        )

        # 3. Retry loop if configured
        max_retries = self.settings.generation.numeric_retry_attempts
        if max_retries > 0:
            strict_query = (
                f"{query}\n\n"
                f"[STRICT CORRECTION: Your previous answer contained ungrounded numbers: "
                f"{', '.join(verification.ungrounded_numbers)}. You MUST ONLY use numbers "
                f"present in the verified records. If unsure, output REFUSE.]"
            )
            retry_result = self.generator.generate(strict_query, chunks)
            if retry_result.is_refusal:
                return retry_result

            retry_verif = self.verifier.verify(retry_result.text, chunks)
            if retry_verif.is_valid:
                logger.info("Guardrail retry succeeded.")
                return retry_result

            logger.warning(
                "Guardrail retry failed: still ungrounded %s",
                retry_verif.ungrounded_numbers,
            )

        # 4. Safe fallback
        return GenerationResult(
            text=STANDARD_REFUSAL,
            is_refusal=True,
            word_count=len(STANDARD_REFUSAL.split()),
            context_chunks=tuple(chunks),
        )

