from __future__ import annotations

import pytest

from lectura.chunking import Line, chunk_transcript
from lectura.config import IndexConfig, RetrievalConfig
from lectura.embeddings import HashEmbedder
from lectura.index import ChromaIndex, Hit
from lectura.retrieval import BM25Retriever, HybridRRF, RetrievalPipeline, tokenize

V1, V2 = "aaaaaaaaaaa", "bbbbbbbbbbb"
TOPICS = ["gradient descent", "backpropagation", "convolution", "attention", "dropout", "batchnorm"]


def _lines(prefix: str) -> list[Line]:
    return [Line(f"{prefix} lecture about {t} explained", i * 30.0, 20.0) for i, t in enumerate(TOPICS)]


@pytest.fixture
def index(settings) -> ChromaIndex:
    idx = ChromaIndex(settings.chroma_dir, IndexConfig(embedding_model="hash"), HashEmbedder())
    idx.add_chunks(chunk_transcript(_lines("first"), 30, V1) + chunk_transcript(_lines("second"), 30, V2))
    return idx


def test_tokenize_drops_stopwords():
    assert tokenize("The gradient of the loss is computed") == ["gradient", "loss", "computed"]


def test_bm25_ranks_lexical_match_first(index):
    bm = BM25Retriever(index)
    hits = bm.retrieve("dropout regularisation", [V1], k=3)
    assert hits and "dropout" in hits[0].chunk.text and hits[0].source == "bm25"
    assert [h.rank for h in hits] == list(range(len(hits)))
    assert bm.retrieve("zzzz qqqq", [V1], k=3) == []       # no lexical overlap -> nothing
    assert bm.retrieve("dropout", [], k=3) == []
    assert bm.retrieve("the of", [V1], k=3) == []          # only stopwords


def test_bm25_cache_invalidation(index):
    bm = BM25Retriever(index)
    assert bm.retrieve("attention", [V1], k=1)
    index.delete_video(V1)
    assert bm.retrieve("attention", [V1], k=1)             # stale cache still answers
    bm.invalidate()
    assert bm.retrieve("attention", [V1], k=1) == []


def test_hybrid_rrf_fuses_and_ranks(index):
    class Fixed:
        def __init__(self, hits):
            self._hits = hits

        def retrieve(self, q, v, k):
            return self._hits[:k]

    chunks = index.all_chunks([V1])
    dense = Fixed([Hit(chunks[0], 0.9, 0), Hit(chunks[1], 0.8, 1)])
    sparse = Fixed([Hit(chunks[1], 5.0, 0), Hit(chunks[2], 4.0, 1)])
    fused = HybridRRF(dense, sparse, rrf_k=60).retrieve("q", [V1], k=3)
    assert [h.chunk.index for h in fused] == [1, 0, 2]     # chunk 1 appears in both lists
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert all(h.source == "hybrid" for h in fused)


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
def test_pipeline_strategies(index, strategy):
    cfg = RetrievalConfig(strategy=strategy, candidate_k=5, top_k=2)
    p = RetrievalPipeline(index, cfg)
    r = p.retrieve("convolution", [V1, V2])
    assert len(r.hits) == 2 and len(r.candidates) <= 5
    assert "convolution" in r.hits[0].chunk.text
    assert r.covered is True
    assert strategy in r.timings
    assert r.best_score == r.candidates[0].score


def test_pipeline_gate(index):
    p = RetrievalPipeline(index, RetrievalConfig(min_score=0.99))
    r = p.retrieve("attention", [V1])
    assert r.hits and r.covered is False
    p = RetrievalPipeline(index, RetrievalConfig(min_score=None))
    assert p.retrieve("attention", [V1]).covered is True
    assert p.retrieve("attention", []).covered is False


def test_pipeline_k_override_and_candidate_floor(index):
    p = RetrievalPipeline(index, RetrievalConfig(candidate_k=2, top_k=1))
    r = p.retrieve("attention", [V1, V2], k=4)
    assert len(r.hits) == 4 and len(r.candidates) == 4  # candidate pool grows to cover k


def test_gate_uses_pre_rerank_score(index):
    """The gate must read the retrieval score, not the reranker's. A cross
    encoder rescales to its own logit range, which would silently invalidate a
    threshold calibrated on cosine similarity."""

    class ShiftReranker:
        def rerank(self, query, hits, k=None):
            return [Hit(h.chunk, h.score - 100.0, r, "rerank") for r, h in enumerate(hits)]

    p = RetrievalPipeline(index, RetrievalConfig(min_score=0.1))
    p.reranker = ShiftReranker()
    r = p.retrieve("attention", [V1])
    assert r.candidates[0].score < -50          # reranker did rescale
    assert r.gate_score is not None and r.gate_score > 0.1
    assert r.covered is True                    # gate ignored the rescaling
    assert r.best_score == r.candidates[0].score

    p2 = RetrievalPipeline(index, RetrievalConfig(min_score=0.99))
    p2.reranker = ShiftReranker()
    assert p2.retrieve("attention", [V1]).covered is False
