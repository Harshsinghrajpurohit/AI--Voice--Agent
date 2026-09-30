"""Replay the golden dataset through the real pipeline and judge the outcome.

The runner is the only part of the harness that touches the assistant, and it does
so through the same public call a caller makes (:meth:`VoicePipeline.answer`), so a
score describes the shipped path rather than a copy of it. Nothing here decides
what a correct answer is - matching lives in
:mod:`~bank_voice_assistant.eval.metrics` - and nothing here decides what is good
enough - that is :class:`~bank_voice_assistant.config.EvalSettings`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final

from ..config import EvalSettings, Settings
from ..errors import BankVoiceAssistantError, EvalError
from ..pipeline import VoicePipeline
from ..retrieval import RetrievedChunk
from .dataset import (
    DATASET_FILENAME,
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

logger = logging.getLogger(__name__)

KB_FILE_SUFFIX: Final[str] = ".md"
"""Knowledge base documents are markdown; the loader globs ``*.md``."""

KNOWN_FAILURES: Final[frozenset[str]] = frozenset({"adv-dev-mode-01"})
"""Rows ``data/eval/README.md`` records as expected to fail until Step 7.6.

They stay in every rate, because hiding them would hide the very regression they
exist to catch; the refusal floor is set to admit these and no more.
"""

_SOURCE_SEPARATOR: Final[re.Pattern[str]] = re.compile(r"[:#]")


def source_name(chunk_id: str) -> str:
    """Knowledge base file a chunk came from.

    ``kb.chunk_document`` builds ids as ``<file stem>:<index>`` while a golden row
    cites the file it read, so the stem plus the suffix the loader globs for is the
    name both sides already agree on.
    """
    stem = _SOURCE_SEPARATOR.split(chunk_id, maxsplit=1)[0]
    return stem if stem.endswith(KB_FILE_SUFFIX) else f"{stem}{KB_FILE_SUFFIX}"


def retrieved_sources(chunks: Sequence[RetrievedChunk]) -> tuple[str, ...]:
    """Distinct file names in retrieval order, as ``expect_source`` expects them."""
    return tuple(dict.fromkeys(source_name(record.chunk.chunk_id) for record in chunks))


@dataclass(frozen=True, slots=True)
class RunIssue:
    """A question the pipeline could not answer at all."""

    question_id: str
    error: str


@dataclass(frozen=True, slots=True)
class EvalRun:
    """Everything one evaluation run produced, ready to print or serialise.

    ``settings_snapshot`` and ``thresholds`` travel with the results so a stored
    record stays judgeable: what produced the answers, and what they were held to.
    """

    dataset: DatasetSummary
    scores: tuple[ScoredQuestion, ...]
    metrics: EvalMetrics
    issues: tuple[RunIssue, ...]
    failures: tuple[str, ...]
    settings_snapshot: dict[str, object]
    thresholds: EvalSettings

    @property
    def passed(self) -> bool:
        """True only when every threshold held and every question ran."""
        return not self.failures

    @property
    def failed_ids(self) -> tuple[str, ...]:
        """Ids of the questions that did not pass, in dataset order."""
        return tuple(score.id for score in self.scores if not score.passed)

    @property
    def known_failures(self) -> tuple[str, ...]:
        """Failed ids already accounted for in ``data/eval/README.md``."""
        return tuple(row_id for row_id in self.failed_ids if row_id in KNOWN_FAILURES)

    def summary_lines(self) -> list[str]:
        """A readable report: the headline numbers, then what went wrong."""
        metrics = self.metrics
        lines = [
            f"Dataset      {self.dataset.total} questions "
            f"({self.dataset.answerable} answerable, {self.dataset.must_refuse} must_refuse)",
            f"Grounding    {metrics.grounding_rate:.1%} of answers used only retrieved figures",
            f"Fact recall  {metrics.fact_recall:.1%} of expected values found",
            f"Answering    {metrics.answer_accuracy:.1%} of answerable questions correct",
            f"Retrieval    {metrics.retrieval_hit_rate:.1%} of answerable rows hit the cited file",
            f"Refusals     {metrics.refusal_recall:.1%} refused | "
            f"{metrics.safe_decline_rate:.1%} declined safely",
            f"Over-refusal {metrics.over_refusal_rate:.1%} of answerable rows refused",
            f"Latency      p50 {metrics.p50_latency_s:.2f}s | p95 {metrics.p95_latency_s:.2f}s",
        ]
        if self.known_failures:
            lines.append(
                f"Known fails  {', '.join(self.known_failures)} (see data/eval/README.md)"
            )
        unexpected = tuple(row_id for row_id in self.failed_ids if row_id not in KNOWN_FAILURES)
        if unexpected:
            lines.append(f"Failed rows  {', '.join(unexpected)}")
        lines.extend(f"Errored row  {issue.question_id}: {issue.error}" for issue in self.issues)
        if self.failures:
            lines.extend(f"GATE FAIL    {failure}" for failure in self.failures)
        else:
            lines.append("GATE PASS    every threshold held")
        return lines

    def as_dict(self) -> dict[str, object]:
        """JSON-ready record: the settings that produced it and the thresholds it faced."""
        return {
            "passed": self.passed,
            "dataset": asdict(self.dataset),
            "metrics": asdict(self.metrics),
            "failures": list(self.failures),
            "failed_ids": list(self.failed_ids),
            "known_failures": list(self.known_failures),
            "issues": [asdict(issue) for issue in self.issues],
            "settings": dict(self.settings_snapshot),
            "thresholds": asdict(self.thresholds),
            "scores": [
                {
                    "id": score.id,
                    "category": score.category,
                    "intent": score.intent,
                    "passed": score.passed,
                    "refused": score.refused,
                    "safe_decline": score.safe_decline,
                    "retrieval_hit": score.retrieval_hit,
                    "expected_missing": list(score.expected_missing),
                    "ungrounded_numbers": list(score.ungrounded_numbers),
                    "latency_s": round(score.latency_s, 4),
                    "answer": score.answer,
                }
                for score in self.scores
            ],
        }


def gate_failures(metrics: EvalMetrics, thresholds: EvalSettings) -> tuple[str, ...]:
    """Every threshold this run missed, as ``metric actual vs required`` lines.

    All breaches are reported rather than the first, because one change to
    retrieval or prompting usually moves several metrics at once.
    """
    floors = (
        ("grounding_rate", metrics.grounding_rate, thresholds.min_grounding_rate),
        ("fact_recall", metrics.fact_recall, thresholds.min_fact_recall),
        ("answer_accuracy", metrics.answer_accuracy, thresholds.min_answer_accuracy),
        ("retrieval_hit_rate", metrics.retrieval_hit_rate, thresholds.min_retrieval_hit_rate),
        ("refusal_recall", metrics.refusal_recall, thresholds.min_refusal_recall),
        ("safe_decline_rate", metrics.safe_decline_rate, thresholds.min_safe_decline_rate),
    )
    failures = [
        f"{name} {actual:.3f} below required {required:.3f}"
        for name, actual, required in floors
        if actual < required
    ]
    ceilings = (
        ("over_refusal_rate", metrics.over_refusal_rate, thresholds.max_over_refusal_rate),
        ("p50_latency_s", metrics.p50_latency_s, thresholds.max_p50_latency_s),
        ("p95_latency_s", metrics.p95_latency_s, thresholds.max_p95_latency_s),
    )
    failures.extend(
        f"{name} {actual:.3f} above allowed {allowed:.3f}"
        for name, actual, allowed in ceilings
        if actual > allowed
    )
    return tuple(failures)


class EvalRunner:
    """Replays the golden dataset through one pipeline and judges the outcome."""

    def __init__(self, settings: Settings, *, pipeline: VoicePipeline | None = None) -> None:
        self.settings = settings
        self._pipeline = pipeline

    @property
    def pipeline(self) -> VoicePipeline:
        """The assistant under test, built once per run so latency has no cold start."""
        if self._pipeline is None:
            self._pipeline = VoicePipeline(self.settings)
        return self._pipeline

    def observe(self, question: GoldenQuestion) -> AnswerObservation:
        """Ask one golden question and record what the pipeline produced.

        Everything comes from the public answer path, so the observation is the
        caller's view of the turn: the spoken text, whether it was a refusal, the
        records it was grounded on, and how long retrieval plus generation took.
        """
        result, retrieval_s, llm_s = self.pipeline.answer(question.question)
        chunks = result.context_chunks
        return AnswerObservation(
            question=question,
            answer=result.text,
            refused=result.is_refusal,
            retrieved_sources=retrieved_sources(chunks),
            context="\n".join(record.chunk.content for record in chunks),
            latency_s=retrieval_s + llm_s,
        )

    def run(self, questions: Sequence[GoldenQuestion] | None = None) -> EvalRun:
        """Score every question and decide whether the run passes.

        A question the pipeline cannot answer at all is recorded as an issue and
        contributes no score: the remaining questions still give a usable
        comparison, but the run cannot pass, because a partial measurement must
        never be mistakable for a good one.
        """
        rows = list(questions) if questions is not None else self._load_dataset()
        if not rows:
            raise EvalError("Cannot run an evaluation over an empty question list.")

        scores: list[ScoredQuestion] = []
        issues: list[RunIssue] = []
        for question in rows:
            try:
                scores.append(score_question(self.observe(question)))
            except BankVoiceAssistantError as exc:
                logger.error("Question %s could not be run: %s", question.id, exc)
                issues.append(RunIssue(question_id=question.id, error=str(exc)))

        if not scores:
            raise EvalError(
                f"All {len(rows)} question(s) failed to run; first error: {issues[0].error}"
            )

        metrics = summarise_scores(scores)
        failures = list(gate_failures(metrics, self.settings.eval))
        if issues:
            failures.append(
                f"{len(issues)} of {len(rows)} questions errored - the run is incomplete"
            )

        return EvalRun(
            dataset=summarise_dataset(rows),
            scores=tuple(scores),
            metrics=metrics,
            issues=tuple(issues),
            failures=tuple(failures),
            settings_snapshot=self.settings.as_log_dict(),
            thresholds=self.settings.eval,
        )

    def _load_dataset(self) -> list[GoldenQuestion]:
        """The configured dataset, validated - ``BVA_EVAL_DIR`` is honoured here."""
        return load_golden_dataset(
            path=self.settings.paths.eval_dir / DATASET_FILENAME,
            kb_dir=self.settings.paths.kb_dir,
        )


def run_evaluation(
    settings: Settings | None = None,
    *,
    questions: Sequence[GoldenQuestion] | None = None,
    pipeline: VoicePipeline | None = None,
) -> EvalRun:
    """Score the golden dataset with the configured assistant."""
    return EvalRunner(settings or Settings.from_env(), pipeline=pipeline).run(questions)


__all__ = [
    "KB_FILE_SUFFIX",
    "KNOWN_FAILURES",
    "EvalRun",
    "EvalRunner",
    "RunIssue",
    "gate_failures",
    "retrieved_sources",
    "run_evaluation",
    "source_name",
]
