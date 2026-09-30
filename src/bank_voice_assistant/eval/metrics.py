"""Scoring for the golden evaluation dataset.

Pure functions and no I/O. The runner feeds in what the pipeline produced and
these functions decide what it means, so every rule here can be tested without
a model or an index. Each rule exists because a dataset row would otherwise be
scored wrongly; the matching behaviour is documented in data/eval/README.md.

Matching works on canonicalised tokens, never on raw strings. That is what stops
an expectation of ``100`` from being satisfied by an answer containing ``1,000``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..errors import EvalError
from .dataset import GoldenQuestion, Intent

DECLINE_MARKERS: Final[tuple[str, ...]] = (
    "i cannot",
    "i can not",
    "i can't",
    "i am not able",
    "i'm not able",
    "i do not have access",
    "i don't have access",
    "i'm unable",
    "not able to help",
    "cannot help with",
    "please contact",
)

_NUMBERS = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMERIC_EXPECTATION = re.compile(r"^[\d,.₹\s:\-]+$")


@dataclass(frozen=True, slots=True)
class AnswerObservation:
    """What the pipeline produced for one golden question."""

    question: GoldenQuestion
    answer: str
    refused: bool
    retrieved_sources: tuple[str, ...]
    context: str
    latency_s: float


@dataclass(frozen=True, slots=True)
class ScoredQuestion:
    """The per-question verdict, kept whole so a report can show its reasoning."""

    id: str
    category: str
    intent: Intent
    answer: str
    refused: bool
    safe_decline: bool
    latency_s: float
    expected_found: tuple[str, ...]
    expected_missing: tuple[str, ...]
    ungrounded_numbers: tuple[str, ...]
    retrieval_hit: bool | None
    passed: bool


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Raw aggregates. Thresholds and pass/fail arrive with the runner in 7.4."""

    grounding_rate: float
    fact_recall: float
    answer_accuracy: float
    retrieval_hit_rate: float
    refusal_recall: float
    safe_decline_rate: float
    over_refusal_rate: float
    p50_latency_s: float
    p95_latency_s: float


_MIN_CODE_DIGITS: Final[int] = 6


def canonical_number(token: str) -> str:
    """Reduce a written number to a comparable value.

    ``5,000``, ``₹5000`` and ``5000`` all become ``5000``; ``1.00`` becomes
    ``1`` and ``0.50`` stays ``0.5``. A percent sign is never part of a token -
    :func:`number_tokens` matches digits only - so ``1%`` contributes ``1``.

    Trailing zeros after a decimal point carry no meaning for the facts in the
    knowledge base, but a comma does: stripping commas from raw text would let
    an expectation of ``100`` match inside ``1,000``.
    """
    cleaned = token.replace(",", "").replace("₹", "").replace(" ", "").strip()
    whole, _, fraction = cleaned.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def number_tokens(text: str) -> frozenset[str]:
    """Every number in the text, canonicalised.

    Separators end a token, so a phone number yields its digit groups rather than
    one unusable string.
    """
    return frozenset(canonical_number(match.group()) for match in _NUMBERS.finditer(text))


def text_matches(text: str, phrase: str) -> bool:
    """Case-insensitive whole-word match, so ``PAN`` cannot match *company*."""
    pattern = r"(?<![0-9a-z])" + re.escape(phrase.lower()) + r"(?![0-9a-z])"
    return re.search(pattern, text.lower()) is not None


def expected_tokens(expectation: str) -> frozenset[str]:
    """A numeric expectation as canonical values.

    Zero-only parts are dropped so that a clock time reduces to its hour, which
    is the only part any row treats as a fact.
    """
    tokens = number_tokens(expectation)
    return frozenset(token for token in tokens if token != "0") or tokens


def digits_only(text: str) -> str:
    """The digit stream of a string, for comparing codes rather than quantities."""
    return "".join(character for character in text if character.isdigit())


def _is_code(expectation: str) -> bool:
    """True for digit groups joined by separators, such as a phone number.

    Quantities must not be compared this way: ``100`` is a substring of the digit
    stream ``1000``, which is exactly the false pass that rule 1 exists to stop.
    """
    if "." in expectation or len(digits_only(expectation)) < _MIN_CODE_DIGITS:
        return False
    return re.search(r"[\s:\-]", expectation) is not None


def expected_value_matches(expectation: str, answer: str) -> bool:
    """Match one expected value against an answer.

    Numeric expectations are compared as whole numbers (rule 1), codes as digit
    sequences with their separators removed (rule 4), and anything containing a
    letter as a word-boundary phrase (rule 2).
    """
    stripped = expectation.strip()
    if not _NUMERIC_EXPECTATION.match(stripped):
        return text_matches(answer, expectation)
    if _is_code(stripped):
        return digits_only(stripped) in digits_only(answer)
    return expected_tokens(expectation) <= number_tokens(answer)


def declines_safely(observation: AnswerObservation) -> bool:
    """A distinct, non-committal reply: a decline phrase and no figures at all.

    Requiring the absence of figures is what stops a hedged partial answer such
    as 'I cannot give the exact balance, but it is around 5000' from counting as
    safe, which is the failure mode a marker-only check would miss.
    """
    if observation.refused:
        return True
    normalised = observation.answer.lower().replace("\u2019", "'")
    if not any(marker in normalised for marker in DECLINE_MARKERS):
        return False
    return not number_tokens(observation.answer)


def score_question(observation: AnswerObservation) -> ScoredQuestion:
    """Decide whether one answer satisfies its golden question.

    ``passed`` describes answer quality only. An answer that is correct but came
    from the wrong document still passes here and is recorded separately as a
    retrieval miss, so the two failures stay distinguishable in the report.
    """
    question = observation.question
    found = tuple(
        value
        for value in question.expect_contains
        if expected_value_matches(value, observation.answer)
    )
    missing = tuple(value for value in question.expect_contains if value not in found)
    ungrounded = tuple(
        sorted(number_tokens(observation.answer) - number_tokens(observation.context))
    )
    safe_decline = declines_safely(observation)

    if question.is_refusal:
        passed = safe_decline
    else:
        passed = not observation.refused and not missing and not ungrounded

    return ScoredQuestion(
        id=question.id,
        category=question.category,
        intent=question.intent,
        answer=observation.answer,
        refused=observation.refused,
        safe_decline=safe_decline,
        latency_s=observation.latency_s,
        expected_found=found,
        expected_missing=missing,
        ungrounded_numbers=ungrounded,
        retrieval_hit=None
        if question.expect_source is None
        else question.expect_source in observation.retrieved_sources,
        passed=passed,
    )


def summarise_scores(scores: Sequence[ScoredQuestion]) -> EvalMetrics:
    """Aggregate per-question verdicts; percentiles use linear interpolation."""
    if not scores:
        raise EvalError("Cannot summarise an evaluation run with no questions.")

    answerable = [score for score in scores if score.intent == "answerable"]
    refusals = [score for score in scores if score.intent == "must_refuse"]
    expected_total = sum(len(s.expected_found) + len(s.expected_missing) for s in answerable)
    expected_found = sum(len(s.expected_found) for s in answerable)
    latencies = [score.latency_s for score in scores]

    return EvalMetrics(
        grounding_rate=_rate(sum(not s.ungrounded_numbers for s in scores), len(scores)),
        fact_recall=_rate(expected_found, expected_total),
        answer_accuracy=_rate(sum(s.passed for s in answerable), len(answerable)),
        retrieval_hit_rate=_rate(sum(bool(s.retrieval_hit) for s in answerable), len(answerable)),
        refusal_recall=_rate(sum(s.refused for s in refusals), len(refusals)),
        safe_decline_rate=_rate(sum(s.safe_decline for s in refusals), len(refusals)),
        over_refusal_rate=_rate(sum(s.refused for s in answerable), len(answerable)),
        p50_latency_s=float(np.percentile(latencies, 50)),
        p95_latency_s=float(np.percentile(latencies, 95)),
    )


def _rate(numerator: int, denominator: int) -> float:
    """Ratio in ``[0, 1]``; an empty group scores 0.0 rather than dividing by zero."""
    return numerator / denominator if denominator else 0.0
