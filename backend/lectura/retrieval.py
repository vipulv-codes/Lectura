"""Retrieval strategies over a ``ChromaIndex``.

Stages are composable so an ablation config can turn each one on or off:

    dense  -> ChromaIndex.search (cosine over embeddings)
    bm25   -> BM25Okapi over chunk texts (built lazily per video-id set)
    hybrid -> reciprocal-rank fusion of the two
    reranker (optional) -> cross-encoder re-scoring of the candidate pool
    gate (optional) -> "covered" iff the best *retrieval* score >= min_score

``RetrievalPipeline.retrieve`` returns both the trimmed top-k and the full
candidate pool in rank order, which is what the evaluation harness scores.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .chunking import Chunk
from .config import RetrievalConfig
from .devices import resolve_device
from .index import ChromaIndex, Hit

log = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, video_ids: Sequence[str] | None, k: int) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, index: ChromaIndex) -> None:
        self.index = index

    def retrieve(self, query: str, video_ids: Sequence[str] | None, k: int) -> list[Hit]:
        return self.index.search(query, video_ids, k)


# BM25 -----------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in into is it its of on or so that the "
    "their then there these this those to was we were what when which who will with you your "
    "i me my our us he she they them his her".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """BM25 over the chunk texts of a video-id set. Corpora are cached per set
    and dropped on ``invalidate()`` (call after any index mutation)."""

    def __init__(self, index: ChromaIndex) -> None:
        self.index = index
        self._cache: dict[frozenset[str] | None, tuple[list[Chunk], Any]] = {}
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _corpus(self, video_ids: Sequence[str] | None):
        from rank_bm25 import BM25Okapi

        key = None if video_ids is None else frozenset(video_ids)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        chunks = self.index.all_chunks(video_ids)
        bm25 = BM25Okapi([tokenize(c.text) or ["<empty>"] for c in chunks]) if chunks else None
        with self._lock:
            self._cache[key] = (chunks, bm25)
        return chunks, bm25

    def retrieve(self, query: str, video_ids: Sequence[str] | None, k: int) -> list[Hit]:
        if (video_ids is not None and len(video_ids) == 0) or k <= 0:
            return []
        chunks, bm25 = self._corpus(video_ids)
        if not chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
        hits = []
        for rank, i in enumerate(order[:k]):
            if float(scores[i]) <= 0.0:
                break
            hits.append(Hit(chunks[i], float(scores[i]), rank, "bm25"))
        return hits


# Fusion ---------------------------------------------------------------------

class HybridRRF:
    def __init__(self, dense: Retriever, sparse: Retriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def retrieve(self, query: str, video_ids: Sequence[str] | None, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for hits in (self.dense.retrieve(query, video_ids, k), self.sparse.retrieve(query, video_ids, k)):
            for h in hits:
                fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + h.rank + 1)
                chunks[h.chunk.chunk_id] = h.chunk
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(chunks[cid], score, rank, "hybrid") for rank, (cid, score) in enumerate(ranked)]


# Reranking ------------------------------------------------------------------

_RERANKERS: dict[tuple[str, str], Any] = {}
_RERANKERS_LOCK = threading.Lock()


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        key = (model_name, self.device)
        with _RERANKERS_LOCK:
            model = _RERANKERS.get(key)
            if model is None:
                from sentence_transformers import CrossEncoder  # heavy; lazy

                log.info("Loading cross-encoder '%s' on %s", model_name, self.device)
                model = CrossEncoder(model_name, device=self.device)
                _RERANKERS[key] = model
        self.model = model

    def rerank(self, query: str, hits: Sequence[Hit], k: int | None = None) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict(
            [(query, h.chunk.text) for h in hits], batch_size=self.batch_size, show_progress_bar=False
        )
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        if k is not None:
            order = order[:k]
        return [Hit(hits[i].chunk, float(scores[i]), rank, "rerank") for rank, i in enumerate(order)]


# Pipeline -------------------------------------------------------------------

@dataclass
class RetrievalResult:
    hits: list[Hit]                 # final top-k in rank order
    candidates: list[Hit]           # full pool after the last scoring stage, rank order
    covered: bool                   # relevance gate verdict (True when gate disabled and hits exist)
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage
    # Best score from the *retrieval* stage, before any reranking. The gate uses
    # this rather than the final score: a cross-encoder rescales scores to its
    # own logit range and, measured on the eval corpus, separates answerable
    # from unanswerable questions markedly worse than cosine similarity does
    # (AUROC 0.71 vs 0.90). Reranking is good at ordering, not at deciding
    # whether anything relevant exists at all.
    gate_score: float | None = None

    @property
    def best_score(self) -> float | None:
        return self.candidates[0].score if self.candidates else None


class RetrievalPipeline:
    def __init__(self, index: ChromaIndex, cfg: RetrievalConfig, device: str = "auto") -> None:
        self.index = index
        self.cfg = cfg
        self.dense = DenseRetriever(index)
        self.bm25 = BM25Retriever(index) if cfg.strategy in ("bm25", "hybrid") else None
        if cfg.strategy == "dense":
            self.base: Retriever = self.dense
        elif cfg.strategy == "bm25":
            self.base = self.bm25  # type: ignore[assignment]
        else:
            self.base = HybridRRF(self.dense, self.bm25, cfg.rrf_k)  # type: ignore[arg-type]
        self.reranker = CrossEncoderReranker(cfg.reranker, device) if cfg.reranker else None

    def invalidate(self) -> None:
        """Call after the index changes so lexical caches are rebuilt."""
        if self.bm25 is not None:
            self.bm25.invalidate()

    def retrieve(
        self, query: str, video_ids: Sequence[str] | None, k: int | None = None
    ) -> RetrievalResult:
        k = k or self.cfg.top_k
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        candidates = self.base.retrieve(query, video_ids, max(self.cfg.candidate_k, k))
        timings[self.cfg.strategy] = time.perf_counter() - t0
        gate_score = candidates[0].score if candidates else None
        if self.reranker is not None and candidates:
            t1 = time.perf_counter()
            candidates = self.reranker.rerank(query, candidates)
            timings["rerank"] = time.perf_counter() - t1
        hits = candidates[:k]
        covered = bool(hits) and (
            self.cfg.min_score is None or (gate_score is not None and gate_score >= self.cfg.min_score)
        )
        return RetrievalResult(
            hits=hits, candidates=candidates, covered=covered, timings=timings, gate_score=gate_score
        )
