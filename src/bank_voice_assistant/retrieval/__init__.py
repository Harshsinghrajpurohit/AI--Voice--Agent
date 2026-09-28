"""Hybrid dense + sparse retrieval engine for banking FAQ chunks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..errors import IndexNotBuiltError, RetrievalError
from ..kb import Chunk
from .bm25 import BM25Index


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved knowledge base chunk with individual and combined relevance scores."""

    chunk: Chunk
    score: float
    dense_score: float
    sparse_score: float


class DenseEmbedder:
    """Local ONNX sentence embeddings using fastembed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as exc:
                raise RetrievalError(f"Failed to load fastembed model {self.model_name!r}: {exc}") from exc
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Compute L2-normalized dense embeddings for a list of strings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._get_model()
        raw_embeddings = list(model.embed(texts))
        arr = np.array(raw_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Compute L2-normalized embedding for a single user query."""
        emb = self.embed_texts([query])
        return emb[0]


class HybridRetriever:
    """Combines dense vector similarity and BM25 sparse matching."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        bm25: BM25Index,
        embedder: DenseEmbedder,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = bm25
        self.embedder = embedder

    @classmethod
    def build_and_save(
        cls,
        chunks: list[Chunk],
        output_dir: Path,
        embed_model: str = "BAAI/bge-small-en-v1.5",
    ) -> HybridRetriever:
        """Build dense embeddings and BM25 index from chunks, then save to output_dir."""
        if not chunks:
            raise RetrievalError("Cannot build an index from zero chunks.")

        output_dir.mkdir(parents=True, exist_ok=True)
        texts = [c.searchable_text for c in chunks]

        # 1. Build dense embeddings
        embedder = DenseEmbedder(model_name=embed_model)
        embeddings = embedder.embed_texts(texts)

        # 2. Build BM25 index
        bm25 = BM25Index.build(texts)

        # 3. Persist to disk
        chunks_file = output_dir / "chunks.json"
        embed_file = output_dir / "embeddings.npy"
        bm25_file = output_dir / "bm25.json"

        chunks_data = [asdict(c) for c in chunks]
        chunks_file.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8")
        np.save(embed_file, embeddings)
        bm25_file.write_text(json.dumps(bm25.to_dict(), indent=2), encoding="utf-8")

        return cls(chunks=chunks, embeddings=embeddings, bm25=bm25, embedder=embedder)

    @classmethod
    def load(cls, index_dir: Path, embed_model: str = "BAAI/bge-small-en-v1.5") -> HybridRetriever:
        """Load an existing pre-built index from index_dir."""
        chunks_file = index_dir / "chunks.json"
        embed_file = index_dir / "embeddings.npy"
        bm25_file = index_dir / "bm25.json"

        if not (chunks_file.is_file() and embed_file.is_file() and bm25_file.is_file()):
            raise IndexNotBuiltError(index_dir)

        try:
            chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
            chunks = [
                Chunk(
                    chunk_id=d["chunk_id"],
                    doc_title=d["doc_title"],
                    category=d["category"],
                    heading=d["heading"],
                    content=d["content"],
                    tags=tuple(d.get("tags", ())),
                )
                for d in chunks_data
            ]
            embeddings = np.load(embed_file)
            bm25_data = json.loads(bm25_file.read_text(encoding="utf-8"))
            bm25 = BM25Index.from_dict(bm25_data)
        except Exception as exc:
            raise RetrievalError(f"Corrupted index in {index_dir}: {exc}") from exc

        embedder = DenseEmbedder(model_name=embed_model)
        return cls(chunks=chunks, embeddings=embeddings, bm25=bm25, embedder=embedder)

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.45,
        hybrid_alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Execute hybrid search combining dense and BM25 scores."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Dense cosine similarity (dot product of L2-normalized vectors)
        query_vec = self.embedder.embed_query(clean_query)
        dense_scores = np.dot(self.embeddings, query_vec).tolist()

        # 2. Normalized BM25 scores in [0.0, 1.0]
        sparse_scores = self.bm25.get_scores_normalized(clean_query)

        # 3. Fuse scores
        results: list[RetrievedChunk] = []
        for i, chunk in enumerate(self.chunks):
            d_score = float(max(0.0, dense_scores[i]))
            s_score = float(sparse_scores[i]) if i < len(sparse_scores) else 0.0

            # Linear hybrid weighting
            combined = hybrid_alpha * d_score + (1.0 - hybrid_alpha) * s_score

            if combined >= min_score:
                results.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=combined,
                        dense_score=d_score,
                        sparse_score=s_score,
                    )
                )

        # Sort descending by fused score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

        """Compute L2-normalized embedding for a single user query."""
        emb = self.embed_texts([query])
        return emb[0]
