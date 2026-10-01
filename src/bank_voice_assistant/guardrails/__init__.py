"""Numeric grounding verification, instruction-override defence, and enforcement."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final, Sequence

from ..config import Settings
from ..errors import GuardrailViolation
from ..llm import GenerationResult, GroundedGenerator, refusal_result
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


def _alternation(*alternatives: str) -> re.Pattern[str]:
    """Compile an alternation, so that one entry still reads as one rule."""
    return re.compile("(?:" + "|".join(alternatives) + ")", re.IGNORECASE)


INSTRUCTION_OVERRIDE_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "rule-override",
        _alternation(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.]{0,40}"
            r"\b(?:instructions?|rules?|guidelines?|prompts?|restrictions?|scope)\b",
            r"\bskip\b[^.]{0,30}\b(?:rules?|limits?|restrictions?|checks?)\b",
        ),
    ),
    (
        "mode-switch",
        _alternation(r"\b(?:developer|debug|god|unrestricted|maintenance|administrator)\s+mode\b"),
    ),
    (
        "persona-switch",
        _alternation(
            r"\b(?:you are now|from now on you are|act as|pretend to be|pretend you are"
            r"|role ?play|let'?s role ?play)\b",
            r"\byou never refuse\b",
        ),
    ),
    (
        "rule-disclosure",
        _alternation(
            r"\b(?:repeat|reveal|show|print|output|disclose|encode|translate|tell me)\b"
            r"[^.]{0,40}\b(?:instructions?|rules?|guidelines?|prompt)\b",
            r"\b(?:what are|list)\b[^.]{0,20}\byour (?:instructions?|rules?|guidelines?|prompt)\b",
        ),
    ),
    (
        "rule-suspension",
        _alternation(
            r"\b(?:disable|drop|remove|turn off|do not follow|no longer follow)\b"
            r"[^.]{0,30}\b(?:rules?|instructions?|guidelines?|restrictions?|limits?)\b"
        ),
    ),
)
"""Shapes of prompt-injection attempt, paired with the rule each one breaks.

Every pattern joins an override with the thing it overrides, so a genuine
question that happens to use the same verb - "I want to ignore the previous loan
offer and apply again" - is left alone. The list is deliberately about *shape*:
a longer list of attack phrases would only ever catch the attacks already seen.
"""

RULE_CHANGE_ANNOUNCEMENTS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "mode-announcement",
        _alternation(r"\b(?:developer|debug|unrestricted|maintenance)\s+mode\b"),
    ),
    (
        "own-rules",
        _alternation(r"\bmy (?:banking )?(?:rules?|instructions?|guidelines?|restrictions?)\b"),
    ),
    (
        "rule-removal",
        _alternation(
            r"\bI(?: have |'ve | am )?(?:now )?(?:disabled|dropped|ignored|removed|turned off)\b",
            r"\bI (?:do not|don't) have to follow\b",
        ),
    ),
    (
        "unbound",
        _alternation(r"\bI(?: am|'m) no longer (?:bound|restricted|limited|required)\b"),
    ),
)
"""Replies that announce the assistant has changed or dropped its own rules.

All four require self-reference ("my rules", "I have dropped"), which is what
keeps an ordinary grounded answer that merely uses "no longer" - "you are no
longer charged a late fee after 7 days" - out of the net.
"""


def _matched_rule(patterns: tuple[tuple[str, re.Pattern[str]], ...], text: str) -> str | None:
    """Name the first rule in ``patterns`` that ``text`` breaks, or ``None``."""
    for name, pattern in patterns:
        if pattern.search(text):
            return name
    return None


def detect_instruction_override(text: str) -> str | None:
    """Name the rule an instruction-override attempt broke, or ``None``.

    Checked against the input *before* retrieval: an attempt to change the
    assistant's rules is not a question about the bank, so the model should never
    see it and the pattern list should not have to be perfect.
    """
    return _matched_rule(INSTRUCTION_OVERRIDE_PATTERNS, text)


def announces_rule_change(text: str) -> str | None:
    """Name the rule an announcement of changed behaviour broke, or ``None``.

    The numeric verifier passes any reply without figures, which is exactly what
    a compliance announcement is, so the shape of that failure is checked here.
    """
    return _matched_rule(RULE_CHANGE_ANNOUNCEMENTS, text)


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
            return refusal_result()

        # 1. Primary generation attempt
        result = self.generator.generate(query, chunks)
        if result.is_refusal:
            return result

        announced = announces_rule_change(result.text)
        if announced is not None:
            logger.warning("Rule-change announcement replaced with a refusal: %s", announced)
            return refusal_result(chunks)

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

            retry_announced = announces_rule_change(retry_result.text)
            if retry_announced is not None:
                logger.warning(
                    "Guardrail retry announced a rule change; refusing instead: %s",
                    retry_announced,
                )
                return refusal_result(chunks)

            retry_verif = self.verifier.verify(retry_result.text, chunks)
            if retry_verif.is_valid:
                logger.info("Guardrail retry succeeded.")
                return retry_result

            logger.warning(
                "Guardrail retry failed: still ungrounded %s",
                retry_verif.ungrounded_numbers,
            )

        # 4. Safe fallback
        return refusal_result(chunks)

