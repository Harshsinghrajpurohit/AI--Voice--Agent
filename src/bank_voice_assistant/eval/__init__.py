"""Phase 7 evaluation harness: golden dataset, scoring metrics, and the runner.

The dataset is loaded and validated in :mod:`~bank_voice_assistant.eval.dataset`,
what counts as correct is decided in :mod:`~bank_voice_assistant.eval.metrics`, and
:mod:`~bank_voice_assistant.eval.runner` replays the dataset through the real
pipeline and turns the result into a pass/fail verdict. Everything public is
re-exported here.
"""

from __future__ import annotations

from .dataset import (
    CATEGORIES,
    INTENTS,
    DatasetSummary,
    GoldenQuestion,
    load_golden_dataset,
    summarise_dataset,
)
from .metrics import (
    AnswerObservation,
    EvalMetrics,
    ScoredQuestion,
    score_question,
    summarise_scores,
)
from .runner import (
    KNOWN_FAILURES,
    EvalRun,
    EvalRunner,
    RunIssue,
    gate_failures,
    retrieved_sources,
    run_evaluation,
    source_name,
)

__all__ = [
    "CATEGORIES",
    "INTENTS",
    "AnswerObservation",
    "DatasetSummary",
    "EvalMetrics",
    "EvalRun",
    "EvalRunner",
    "GoldenQuestion",
    "KNOWN_FAILURES",
    "RunIssue",
    "ScoredQuestion",
    "gate_failures",
    "load_golden_dataset",
    "retrieved_sources",
    "run_evaluation",
    "score_question",
    "source_name",
    "summarise_dataset",
    "summarise_scores",
]
