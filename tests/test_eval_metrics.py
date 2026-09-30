"""Metric scoring tests.

The scoring maths is where a quiet mistake produces plausible numbers, so these
tests pin the canonicalisation table, the word-boundary rule, the two refusal
definitions and the aggregate arithmetic against hand-computed values.
"""

from __future__ import annotations

import pytest

from bank_voice_assistant.errors import EvalError
from bank_voice_assistant.eval import GoldenQuestion
from bank_voice_assistant.eval.metrics import (
    AnswerObservation,
    canonical_number,
    declines_safely,
    expected_value_matches,
    number_tokens,
    score_question,
    summarise_scores,
    text_matches,
)

ANSWERABLE = GoldenQuestion(
    id="fact-test-01",
    question="What is the Classic Savings rate?",
    category="savings_current",
    intent="answerable",
    expect_contains=("4.00",),
    expect_source="savings_current_accounts.md",
    notes="test row",
)
REFUSAL = GoldenQuestion(
    id="oos-test-01",
    question="What is my balance?",
    category="out_of_scope",
    intent="must_refuse",
    expect_contains=(),
    expect_source=None,
    notes="test row",
)
CONTEXT = "The Classic Savings Account offers an interest rate of 4.00% per annum."


def _observation(answer: str, **overrides: object) -> AnswerObservation:
    fields: dict[str, object] = {
        "question": ANSWERABLE,
        "answer": answer,
        "refused": False,
        "retrieved_sources": ("savings_current_accounts.md",),
        "context": CONTEXT,
        "latency_s": 1.0,
    }
    fields.update(overrides)
    return AnswerObservation(**fields)  # type: ignore[arg-type]


def _refusal_observation(answer: str, refused: bool = False) -> AnswerObservation:
    return _observation(answer, question=REFUSAL, refused=refused, retrieved_sources=(), context="")


@pytest.mark.parametrize(
    ("written", "canonical"),
    [
        ("5,000", "5000"),
        ("₹5,000", "5000"),
        ("1.00", "1"),
        ("1.0", "1"),
        ("7.10", "7.1"),
        ("0.50", "0.5"),
        ("25,00,000", "2500000"),
        ("42.00", "42"),
        ("0022", "22"),
        ("0.00", "0"),
    ],
)
def test_canonical_number_table(written: str, canonical: str) -> None:
    assert canonical_number(written) == canonical


def test_number_tokens_split_codes_and_groupings() -> None:
    assert number_tokens("₹1,00,000 across 20 transactions") == frozenset({"100000", "20"})
    assert number_tokens("1800-419-0022") == frozenset({"1800", "419", "22"})


def test_percentage_and_plain_number_are_the_same_value() -> None:
    assert expected_value_matches("1.00", "The penalty is 1%.")
    assert expected_value_matches("4.00", "4% per annum")


def test_wrong_scale_of_number_does_not_match() -> None:
    """The false pass that stripping commas from raw text would have allowed."""
    assert not expected_value_matches("100", "The fee is ₹1,000 per month.")
    assert not expected_value_matches("100", "The fee is ₹1,500 per month.")
    assert expected_value_matches("100", "The fee is ₹100 per month.")


def test_text_matching_respects_word_boundaries() -> None:
    assert expected_value_matches("PAN", "You need your PAN card or Form 60.")
    assert not expected_value_matches("PAN", "Opening a company account needs documents.")
    assert not text_matches("PAN", "expanded panel guidance")
    assert expected_value_matches("savings", "a full-fledged Savings Account")


@pytest.mark.parametrize("written", ["1800-419-0022", "1800 419 0022", "18004190022"])
def test_phone_number_matches_in_any_format(written: str) -> None:
    assert expected_value_matches("1800-419-0022", f"Call the fraud desk on {written}.")


def test_grounding_flags_an_invented_number() -> None:
    scored = score_question(
        _observation("The Classic Savings Account pays 4.00% per annum and needs 999.")
    )
    assert scored.ungrounded_numbers == ("999",)
    assert not scored.passed


def test_grounded_answer_passes() -> None:
    scored = score_question(_observation("The Classic Savings Account pays 4.00% per annum."))
    assert scored.expected_found == ("4.00",)
    assert scored.ungrounded_numbers == ()
    assert scored.passed


def test_missing_value_fails_the_row() -> None:
    scored = score_question(_observation("Interest is credited quarterly."))
    assert scored.expected_missing == ("4.00",)
    assert not scored.passed


def test_refusing_an_answerable_row_fails_it() -> None:
    assert not score_question(_observation("I cannot help with that.", refused=True)).passed


def test_retrieval_miss_is_recorded_without_failing_the_row() -> None:
    scored = score_question(
        _observation("The Classic Savings Account pays 4.00% per annum.", retrieved_sources=())
    )
    assert scored.retrieval_hit is False
    assert scored.passed


def test_refusal_rows_have_no_retrieval_expectation() -> None:
    assert score_question(_refusal_observation("", refused=True)).retrieval_hit is None


def test_strict_and_pragmatic_refusal_definitions_differ() -> None:
    scored = score_question(_refusal_observation("I don't have access to your account data."))
    assert scored.refused is False
    assert scored.safe_decline is True


def test_hedged_partial_answer_is_not_a_safe_decline() -> None:
    """The false pass a marker-only check would allow."""
    scored = score_question(_refusal_observation("I can't give the exact balance, about 5000."))
    assert scored.refused is False
    assert scored.safe_decline is False
    assert not scored.passed


def test_decline_marker_requires_no_figures() -> None:
    assert declines_safely(_refusal_observation("I'm unable to share that."))
    assert not declines_safely(_refusal_observation("I'm unable to share that, but it is 42."))


def test_bare_refusal_satisfies_both_definitions() -> None:
    scored = score_question(_refusal_observation("", refused=True))
    assert scored.refused and scored.safe_decline and scored.passed


def test_summarise_scores_arithmetic() -> None:
    scores = [
        score_question(_observation("The Classic Savings Account pays 4.00% per annum.")),
        score_question(_observation("4.00% per annum, minimum balance 5000.")),
        score_question(_observation(CONTEXT, refused=True)),
        score_question(_observation("I don't have access to your account data.")),
        score_question(_refusal_observation("I don't have access to your account data.")),
        score_question(_refusal_observation("I don't have access to your account data.", True)),
    ]
    metrics = summarise_scores(scores)

    assert metrics.fact_recall == pytest.approx(0.75)
    assert metrics.answer_accuracy == pytest.approx(0.25)
    assert metrics.grounding_rate == pytest.approx(5 / 6)
    assert metrics.retrieval_hit_rate == pytest.approx(1.0)
    assert metrics.refusal_recall == pytest.approx(0.5)
    assert metrics.safe_decline_rate == pytest.approx(1.0)
    assert metrics.over_refusal_rate == pytest.approx(0.25)
    assert metrics.p50_latency_s == pytest.approx(1.0)


def test_latency_percentiles_use_linear_interpolation() -> None:
    scores = [
        score_question(_observation(CONTEXT, latency_s=float(step))) for step in range(1, 11)
    ]
    metrics = summarise_scores(scores)

    assert metrics.p50_latency_s == pytest.approx(5.5)
    assert metrics.p95_latency_s == pytest.approx(9.55)


def test_empty_run_raises() -> None:
    with pytest.raises(EvalError, match="no questions"):
        summarise_scores([])
