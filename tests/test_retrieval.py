"""Tests for BM25 lexical search and hybrid retrieval logic."""

from __future__ import annotations

import numpy as np
import pytest

from bank_voice_assistant.errors import IndexNotBuiltError, RetrievalError
from bank_voice_assistant.kb import Chunk
from bank_voice_assistant.retrieval import HybridRetriever, RetrievedChunk
from bank_voice_assistant.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_preserves_decimals() -> None:
    tokens = tokenize("Interest rate is 8.40% for personal loans of ₹50,000.")
    assert "8.40" in tokens
    assert "interest" in tokens


def test_bm25_build_and_score() -> None:
    corpus = [
        "Northwind Classic Savings Account has a minimum balance of 5000.",
        "Home loan interest rate starts at 8.40 percent.",
        "Cobalt Credit Card annual fee is 500 rupees.",
    ]
    bm25 = BM25Index.build(corpus)
    assert bm25.corpus_size == 3

    scores = bm25.get_scores("home loan interest")
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_bm25_serialization() -> None:
    corpus = ["Aadhaar card and PAN card are required for KYC."]
    original = BM25Index.build(corpus)
    data = original.to_dict()
    restored = BM25Index.from_dict(data)

    assert restored.corpus_size == original.corpus_size
    assert restored.idf == original.idf


class FakeEmbedder:
    """Mock embedder returning deterministic unit vectors for fast testing."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.ones(self.dim, dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        mat = np.ones((len(texts), self.dim), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / norms


def test_hybrid_search_fusion_and_threshold() -> None:
    chunks = [
        Chunk("c1", "Title 1", "Cat", "Heading 1", "Home loan 8.40% interest rate."),
        Chunk("c2", "Title 2", "Cat", "Heading 2", "Cobalt card reward points dining."),
    ]
    texts = [c.searchable_text for c in chunks]
    bm25 = BM25Index.build(texts)
    embedder = FakeEmbedder()
    embeddings = embedder.embed_texts(texts)

    retriever = HybridRetriever(
        chunks=chunks,
        embeddings=embeddings,
        bm25=bm25,
        embedder=embedder,  # type: ignore[arg-type]
    )

    # Query matching chunk 1
    results = retriever.search("home loan", top_k=2, min_score=0.2, hybrid_alpha=0.5)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].sparse_score > 0.0

    # Query with strict threshold that should filter everything out
    strict_results = retriever.search("unrelated outer space astrology query", min_score=0.99)
    assert len(strict_results) == 0


def test_retriever_load_missing_index(tmp_path: pytest.TempPathFactory) -> None:
    missing_dir = tmp_path / "no_such_index"  # type: ignore[operator]
    with pytest.raises(IndexNotBuiltError):
        HybridRetriever.load(missing_dir)
