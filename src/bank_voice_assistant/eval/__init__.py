"""Phase 7 evaluation harness: golden dataset, scoring metrics, and CLI runner.

The dataset is loaded and validated in :mod:`~bank_voice_assistant.eval.dataset`.
Metrics and the runner are added by later steps of Phase 7 and re-exported here
as they land.
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

__all__ = [
    "CATEGORIES",
    "INTENTS",
    "DatasetSummary",
    "GoldenQuestion",
    "load_golden_dataset",
    "summarise_dataset",
]
