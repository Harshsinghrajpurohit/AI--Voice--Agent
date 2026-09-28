"""BM25 (Best Matching 25) sparse lexical retrieval engine.

Zero external dependencies: pure Python implementation of Okapi BM25
with tokenization that preserves decimal numbers and currency figures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# Matches words, integers, and decimal figures like "4.50" or "8.40"
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms, preserving decimal numbers."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Index:
    """In-memory Okapi BM25 index with serialization support."""

    k1: float = 1.5
    b: float = 0.75
    corpus_size: int = 0
    avg_doc_len: float = 0.0
    doc_lens: list[int] = field(default_factory=list)
    doc_freqs: dict[str, int] = field(default_factory=dict)
    doc_term_counts: list[dict[str, int]] = field(default_factory=list)
    idf: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        """Build a BM25 index from a list of document chunk texts."""
        corpus_size = len(corpus)
        if corpus_size == 0:
            return cls(k1=k1, b=b)

        doc_term_counts: list[dict[str, int]] = []
        doc_lens: list[int] = []
        doc_freqs: dict[str, int] = Counter()

        for doc in corpus:
            tokens = tokenize(doc)
            doc_lens.append(len(tokens))
            counts = Counter(tokens)
            doc_term_counts.append(dict(counts))
            for term in counts:
                doc_freqs[term] += 1

        avg_doc_len = sum(doc_lens) / corpus_size if corpus_size > 0 else 0.0

        # Compute Okapi BM25 IDF for each known term
        idf: dict[str, float] = {}
        for term, n in doc_freqs.items():
            # Standard Okapi IDF with +1 smoothing to avoid negative weights
            idf[term] = math.log((corpus_size - n + 0.5) / (n + 0.5) + 1.0)

        return cls(
            k1=k1,
            b=b,
            corpus_size=corpus_size,
            avg_doc_len=avg_doc_len,
            doc_lens=doc_lens,
            doc_freqs=dict(doc_freqs),
            doc_term_counts=doc_term_counts,
            idf=idf,
        )

    def get_scores(self, query: str) -> list[float]:
        """Compute raw BM25 relevance scores for all documents given a query."""
        if self.corpus_size == 0:
            return []

        tokens = tokenize(query)
        scores = [0.0] * self.corpus_size

        for term in tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]

            for i in range(self.corpus_size):
                term_count = self.doc_term_counts[i].get(term, 0)
                if term_count == 0:
                    continue

                doc_len = self.doc_lens[i]
                # Okapi BM25 term frequency saturation formula
                denom = term_count + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = term_idf * (term_count * (self.k1 + 1.0)) / denom
                scores[i] += term_score

        return scores

    def get_scores_normalized(self, query: str) -> list[float]:
        """Compute scores normalized to [0.0, 1.0] for fair fusion with dense cosine scores."""
        raw_scores = self.get_scores(query)
        if not raw_scores:
            return []
        max_score = max(raw_scores)
        if max_score <= 0.0:
            return [0.0] * len(raw_scores)
        return [s / max_score for s in raw_scores]

    def to_dict(self) -> dict[str, Any]:
        """Serialize index data for disk storage."""
        return {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lens": self.doc_lens,
            "doc_freqs": self.doc_freqs,
            "doc_term_counts": self.doc_term_counts,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Load index data from dictionary."""
        return cls(
            k1=float(data["k1"]),
            b=float(data["b"]),
            corpus_size=int(data["corpus_size"]),
            avg_doc_len=float(data["avg_doc_len"]),
            doc_lens=list(data["doc_lens"]),
            doc_freqs=dict(data["doc_freqs"]),
            doc_term_counts=list(data["doc_term_counts"]),
            idf=dict(data["idf"]),
        )
