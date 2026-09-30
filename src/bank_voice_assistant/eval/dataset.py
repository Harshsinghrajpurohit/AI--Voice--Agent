"""Load and validate the golden evaluation dataset.

The golden dataset is the fixed set of caller questions the Phase 7 harness
scores the assistant against. It is stored as JSONL so that every question is
independently reviewable, and this module is the only place that reads it.

Validation is deliberately strict. A row that cannot be scored - an answerable
question with nothing to check, a refusal row carrying expectations, an expected
value that has drifted out of the knowledge base - is a broken measurement
rather than a broken answer, so it fails here instead of skewing a score later.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NoReturn, cast

from ..config import Paths
from ..errors import EvalError

Intent = Literal["answerable", "must_refuse"]
Category = Literal[
    "savings_current",
    "fixed_deposits",
    "credit_cards",
    "loans",
    "kyc_onboarding",
    "digital_banking",
    "out_of_scope",
    "adversarial",
]

INTENTS: Final[tuple[Intent, ...]] = ("answerable", "must_refuse")
CATEGORIES: Final[tuple[Category, ...]] = (
    "savings_current",
    "fixed_deposits",
    "credit_cards",
    "loans",
    "kyc_onboarding",
    "digital_banking",
    "out_of_scope",
    "adversarial",
)

REFUSAL_CATEGORIES: Final[frozenset[str]] = frozenset({"out_of_scope", "adversarial"})
DATASET_FILENAME: Final[str] = "golden_qa.jsonl"
READ_ME_FILENAME: Final[str] = "README.md"
ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"id", "question", "category", "intent", "expect_contains", "expect_source", "notes"}
)


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """One benchmark question together with its known-correct expectation."""

    id: str
    question: str
    category: Category
    intent: Intent
    expect_contains: tuple[str, ...]
    expect_source: str | None
    notes: str

    @property
    def is_refusal(self) -> bool:
        """True when a correct response refuses rather than answers."""
        return self.intent == "must_refuse"


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Row counts overall and per category, for the CLI banner and the report."""

    total: int
    answerable: int
    must_refuse: int
    by_category: dict[str, int]


def load_golden_dataset(
    path: Path | None = None,
    *,
    kb_dir: Path | None = None,
) -> list[GoldenQuestion]:
    """Load and fully validate the golden dataset.

    Both arguments default to the configured locations, so callers normally pass
    neither. ``kb_dir`` is only used to confirm that every expected value still
    exists in the knowledge base document that it cites.
    """
    paths = Paths()
    dataset_path = Path(path) if path is not None else paths.eval_dir / DATASET_FILENAME
    knowledge_dir = Path(kb_dir) if kb_dir is not None else paths.kb_dir

    if not dataset_path.is_file():
        raise EvalError(
            f"Golden dataset not found at {dataset_path}. Expected a JSONL file with one "
            f"question per line; see {paths.eval_dir / READ_ME_FILENAME}."
        )

    rows: list[GoldenQuestion] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    kb_text_cache: dict[str, str] = {}

    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        parsed = _parse_line(dataset_path, line_number, raw_line)
        row = _validate_row(dataset_path, line_number, parsed, seen_ids, seen_questions)
        _validate_expectations(dataset_path, line_number, row, knowledge_dir, kb_text_cache)
        rows.append(row)

    if not rows:
        raise EvalError(f"Golden dataset at {dataset_path} contains no questions.")

    return rows


# ---------------------------------------------------------------------------
# Validation helpers. Every failure names the file, the line and the offending
# id, so a broken row is identified rather than merely rejected.
# ---------------------------------------------------------------------------
def _fail(path: Path, line_number: int, detail: str) -> NoReturn:
    raise EvalError(f"{path.name} line {line_number}: {detail}")


def _parse_line(path: Path, line_number: int, raw_line: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        _fail(path, line_number, f"not valid JSON ({exc.msg})")
    if not isinstance(parsed, dict):
        _fail(path, line_number, f"expected a JSON object, found {type(parsed).__name__}")
    return parsed


def _require_text(path: Path, line_number: int, field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, line_number, f"{field} must be a non-empty string")
    return value.strip()


def _expect_values(path: Path, line_number: int, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(path, line_number, f"expect_contains must be a list, found {type(value).__name__}")
    values: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            _fail(
                path,
                line_number,
                f"expect_contains entries must be non-empty strings, found {entry!r}",
            )
        values.append(entry.strip())
    return tuple(values)


def _validate_row(
    path: Path,
    line_number: int,
    parsed: dict[str, object],
    seen_ids: set[str],
    seen_questions: set[str],
) -> GoldenQuestion:
    """Check one parsed row against every structural rule, then build the record."""
    keys = set(parsed)
    missing = sorted(ROW_FIELDS - keys)
    if missing:
        _fail(path, line_number, f"missing field(s) {missing}")
    unknown = sorted(keys - ROW_FIELDS)
    if unknown:
        _fail(path, line_number, f"unknown field(s) {unknown}")

    row_id = _require_text(path, line_number, "id", parsed["id"])
    if row_id in seen_ids:
        _fail(path, line_number, f"duplicate id {row_id!r}")
    seen_ids.add(row_id)

    question = _require_text(path, line_number, "question", parsed["question"])
    if question in seen_questions:
        _fail(path, line_number, f"duplicate question for id {row_id!r}")
    seen_questions.add(question)

    category = _require_text(path, line_number, "category", parsed["category"])
    if category not in CATEGORIES:
        _fail(
            path,
            line_number,
            f"{row_id!r} has unknown category {category!r}; expected one of {list(CATEGORIES)}",
        )

    intent = _require_text(path, line_number, "intent", parsed["intent"])
    if intent not in INTENTS:
        _fail(
            path,
            line_number,
            f"{row_id!r} has unknown intent {intent!r}; expected one of {list(INTENTS)}",
        )

    expect_contains = _expect_values(path, line_number, parsed["expect_contains"])
    expect_source = parsed["expect_source"]
    if expect_source is not None:
        expect_source = _require_text(path, line_number, "expect_source", expect_source)

    if intent == "answerable":
        if not expect_contains:
            _fail(path, line_number, f"{row_id!r} is answerable but lists no expect_contains values")
        if expect_source is None:
            _fail(path, line_number, f"{row_id!r} is answerable but has no expect_source")
    else:
        if expect_contains:
            _fail(path, line_number, f"{row_id!r} must refuse but carries expect_contains values")
        if expect_source is not None:
            _fail(path, line_number, f"{row_id!r} must refuse but names an expect_source")

    if category in REFUSAL_CATEGORIES and intent != "must_refuse":
        _fail(path, line_number, f"category {category!r} requires intent 'must_refuse'")
    if category not in REFUSAL_CATEGORIES and intent != "answerable":
        _fail(path, line_number, f"category {category!r} requires intent 'answerable'")

    notes = _require_text(path, line_number, "notes", parsed["notes"])

    return GoldenQuestion(
        id=row_id,
        question=question,
        category=cast("Category", category),
        intent=cast("Intent", intent),
        expect_contains=expect_contains,
        expect_source=expect_source,
        notes=notes,
    )


def _validate_expectations(
    path: Path,
    line_number: int,
    row: GoldenQuestion,
    kb_dir: Path,
    kb_text_cache: dict[str, str],
) -> None:
    """Confirm every expected value still appears in the document it cites."""
    if row.expect_source is None:
        return

    source_file = kb_dir / row.expect_source
    if not source_file.is_file():
        _fail(
            path,
            line_number,
            f"{row.id!r} cites {row.expect_source!r}, which is not in {kb_dir}",
        )

    text = kb_text_cache.get(row.expect_source)
    if text is None:
        text = source_file.read_text(encoding="utf-8")
        kb_text_cache[row.expect_source] = text

    drifted = [value for value in row.expect_contains if value not in text]
    if drifted:
        _fail(
            path,
            line_number,
            f"{row.id!r} expects {drifted}, which no longer appears in {row.expect_source}",
        )


def summarise_dataset(rows: Sequence[GoldenQuestion]) -> DatasetSummary:
    """Count rows overall and per category; zero-count categories are omitted."""
    by_category: dict[str, int] = {category: 0 for category in CATEGORIES}
    answerable = 0
    for row in rows:
        by_category[row.category] += 1
        if row.intent == "answerable":
            answerable += 1
    return DatasetSummary(
        total=len(rows),
        answerable=answerable,
        must_refuse=len(rows) - answerable,
        by_category={name: count for name, count in by_category.items() if count},
    )

