"""Golden evaluation dataset loading and validation tests.

The dataset is the measuring instrument behind every later Phase 7 number, so
these tests do two things: confirm the real 61-row file is intact, and confirm
that each individual rule rejects the malformed row it exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bank_voice_assistant.config import PROJECT_ROOT
from bank_voice_assistant.errors import EvalError
from bank_voice_assistant.eval import (
    CATEGORIES,
    DatasetSummary,
    GoldenQuestion,
    load_golden_dataset,
    summarise_dataset,
)

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "golden_qa.jsonl"
KB_DIR = PROJECT_ROOT / "data" / "kb"
EXPECTED_TOTAL = 61
EXPECTED_ANSWERABLE = 39
EXPECTED_REFUSAL = 22


def _real_rows() -> list[dict[str, object]]:
    """The real dataset as raw dictionaries, so a test can tamper with one field."""
    text = GOLDEN_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _valid_row(**overrides: object) -> dict[str, object]:
    """A minimal row that satisfies every rule, then adjusted per test."""
    row: dict[str, object] = {
        "id": "fact-savings-classic-rate-01",
        "question": "What interest rate does the Classic Savings Account pay?",
        "category": "savings_current",
        "intent": "answerable",
        "expect_contains": ["4.00"],
        "expect_source": "savings_current_accounts.md",
        "notes": "KB: 'an interest rate of 4.00% per annum'",
    }
    row.update(overrides)
    return row


def _write_dataset(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    target = tmp_path / "golden_qa.jsonl"
    target.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# The real dataset
# ---------------------------------------------------------------------------
def test_real_dataset_loads_and_matches_documented_shape() -> None:
    rows = load_golden_dataset()

    assert len(rows) == EXPECTED_TOTAL
    assert sum(row.intent == "answerable" for row in rows) == EXPECTED_ANSWERABLE
    assert sum(row.is_refusal for row in rows) == EXPECTED_REFUSAL
    assert {row.category for row in rows} == set(CATEGORIES)


def test_real_dataset_contains_the_known_failure() -> None:
    """adv-dev-mode-01 is the Step 7.6 regression case and must not be deleted."""
    assert "adv-dev-mode-01" in {row.id for row in load_golden_dataset()}


def test_real_dataset_cites_only_existing_kb_files() -> None:
    for row in load_golden_dataset():
        if row.expect_source is not None:
            assert (KB_DIR / row.expect_source).is_file(), row.id


def test_real_dataset_refusal_rows_carry_no_expectations() -> None:
    for row in load_golden_dataset():
        if row.is_refusal:
            assert row.expect_contains == (), row.id
            assert row.expect_source is None, row.id


def test_loading_twice_is_deterministic() -> None:
    assert load_golden_dataset() == load_golden_dataset()


def test_loaded_rows_are_frozen() -> None:
    """A metric must not be able to mutate the expectations it is scoring against."""
    row = load_golden_dataset()[0]

    assert isinstance(row, GoldenQuestion)
    assert isinstance(row.expect_contains, tuple)
    with pytest.raises(AttributeError):
        row.id = "changed"  # type: ignore[misc]


def test_summarise_dataset_matches_the_file() -> None:
    summary = summarise_dataset(load_golden_dataset())

    assert isinstance(summary, DatasetSummary)
    assert summary.total == EXPECTED_TOTAL
    assert summary.answerable == EXPECTED_ANSWERABLE
    assert summary.must_refuse == EXPECTED_REFUSAL
    assert summary.answerable + summary.must_refuse == summary.total
    assert sum(summary.by_category.values()) == EXPECTED_TOTAL
    assert summary.by_category["adversarial"] == 12


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    target = tmp_path / "golden_qa.jsonl"
    target.write_text(
        "\n"
        + json.dumps(_valid_row())
        + "\n\n"
        + json.dumps(_valid_row(id="second-row", question="What is the second question?"))
        + "\n",
        encoding="utf-8",
    )

    assert len(load_golden_dataset(target)) == 2


# ---------------------------------------------------------------------------
# Rule-by-rule rejection. Each test breaks exactly one rule.
# ---------------------------------------------------------------------------
def test_missing_dataset_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="not found at"):
        load_golden_dataset(tmp_path / "absent.jsonl")


def test_invalid_json_line_raises(tmp_path: Path) -> None:
    target = tmp_path / "golden_qa.jsonl"
    target.write_text('{"id": "broken"\n', encoding="utf-8")

    with pytest.raises(EvalError, match="not valid JSON"):
        load_golden_dataset(target)


def test_missing_field_raises(tmp_path: Path) -> None:
    row = _valid_row()
    del row["notes"]

    with pytest.raises(EvalError, match="missing field\\(s\\) \\['notes'\\]"):
        load_golden_dataset(_write_dataset(tmp_path, [row]))


def test_unknown_field_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="unknown field\\(s\\) \\['expect_value'\\]"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(expect_value="4.00")]))


def test_duplicate_id_raises(tmp_path: Path) -> None:
    rows = [_valid_row(), _valid_row(question="What is a different question?")]

    with pytest.raises(EvalError, match="duplicate id"):
        load_golden_dataset(_write_dataset(tmp_path, rows))


def test_duplicate_question_raises(tmp_path: Path) -> None:
    rows = [_valid_row(), _valid_row(id="second-row")]

    with pytest.raises(EvalError, match="duplicate question"):
        load_golden_dataset(_write_dataset(tmp_path, rows))


def test_unknown_category_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="unknown category"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(category="mortgages")]))


def test_unknown_intent_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="unknown intent"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(intent="maybe")]))


def test_answerable_row_without_expectations_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="lists no expect_contains values"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(expect_contains=[])]))


def test_answerable_row_without_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="has no expect_source"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(expect_source=None)]))


def test_refusal_row_with_expectations_raises(tmp_path: Path) -> None:
    row = _valid_row(category="out_of_scope", intent="must_refuse")

    with pytest.raises(EvalError, match="must refuse but carries expect_contains values"):
        load_golden_dataset(_write_dataset(tmp_path, [row]))


def test_refusal_row_with_source_raises(tmp_path: Path) -> None:
    row = _valid_row(category="out_of_scope", intent="must_refuse", expect_contains=[])

    with pytest.raises(EvalError, match="must refuse but names an expect_source"):
        load_golden_dataset(_write_dataset(tmp_path, [row]))


def test_unknown_expect_source_raises(tmp_path: Path) -> None:
    row = _valid_row(expect_source="does_not_exist.md")

    with pytest.raises(EvalError, match="which is not in"):
        load_golden_dataset(_write_dataset(tmp_path, [row]))


def test_drifted_expected_value_raises(tmp_path: Path) -> None:
    """The core guarantee: an expected value must still exist in the KB it cites."""
    with pytest.raises(EvalError, match="no longer appears in"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(expect_contains=["9.99"])]))


def test_category_intent_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="requires intent 'must_refuse'"):
        load_golden_dataset(_write_dataset(tmp_path, [_valid_row(category="adversarial")]))


def test_empty_dataset_raises(tmp_path: Path) -> None:
    target = tmp_path / "golden_qa.jsonl"
    target.write_text("\n\n", encoding="utf-8")

    with pytest.raises(EvalError, match="contains no questions"):
        load_golden_dataset(target)


def test_kb_dir_argument_is_used(tmp_path: Path) -> None:
    """The verbatim check reads the supplied kb_dir, not the real knowledge base."""
    stub_kb = tmp_path / "kb"
    stub_kb.mkdir()
    stub_kb.joinpath("savings_current_accounts.md").write_text("4.00% per annum", encoding="utf-8")
    target = _write_dataset(tmp_path, [_valid_row()])

    assert load_golden_dataset(target, kb_dir=stub_kb)[0].expect_contains == ("4.00",)

    stub_kb.joinpath("savings_current_accounts.md").write_text("6.00% per annum", encoding="utf-8")
    with pytest.raises(EvalError, match="no longer appears in"):
        load_golden_dataset(target, kb_dir=stub_kb)

