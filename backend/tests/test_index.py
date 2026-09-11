from __future__ import annotations

import numpy as np
import pytest

from lectura.chunking import Line, chunk_transcript
from lectura.config import IndexConfig
from lectura.embeddings import HashEmbedder
from lectura.index import ChromaIndex

V1, V2 = "aaaaaaaaaaa", "bbbbbbbbbbb"


def _lines(prefix: str, n: int = 6) -> list[Line]:
    topics = ["gradient descent", "backpropagation", "convolution", "attention", "dropout", "batchnorm"]
    return [Line(f"{prefix} lecture about {topics[i]} explained", i * 30.0, 20.0) for i in range(n)]


@pytest.fixture
def index(settings) -> ChromaIndex:
    cfg = IndexConfig(embedding_model="hash", child_secs=30)
    return ChromaIndex(settings.chroma_dir, cfg, HashEmbedder())


def test_collection_name_is_deterministic_and_valid():
    a = IndexConfig().collection_name()
    b = IndexConfig().collection_name()
    c = IndexConfig(child_secs=15).collection_name()
    d = IndexConfig(ef_search=10).collection_name()
    assert a == b != c
    assert a == d  # ef_search is query-time, not identity
    assert 3 <= len(a) <= 63 and a[0].isalnum() and a[-1].isalnum()


def test_add_search_delete_roundtrip(index):
    c1 = chunk_transcript(_lines("first"), 30, V1)
    c2 = chunk_transcript(_lines("second"), 30, V2)
    assert index.add_chunks(c1 + c2) == 12
    assert len(index) == 12
    assert index.video_ids() == {V1, V2}
    assert index.chunk_count(V1) == 6

    hits = index.search("attention", [V1], k=3)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [0, 1, 2]
    assert hits[0].chunk.video_id == V1 and "attention" in hits[0].chunk.text
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert hits[0].chunk.snippet_start == 90.0 and hits[0].chunk.start == 90.0

    both = index.search("dropout", [V1, V2], k=4)
    assert {h.chunk.video_id for h in both} == {V1, V2}
    assert "dropout" in both[0].chunk.text

    everything = index.search("convolution", None, k=2)
    assert len(everything) == 2

    assert index.search("x", [], k=5) == []
    assert index.search("x", [V1], k=0) == []

    index.delete_video(V1)
    assert index.video_ids() == {V2}
    assert index.search("attention", [V1], k=3) == []


def test_upsert_is_idempotent(index):
    chunks = chunk_transcript(_lines("first"), 30, V1)
    index.add_chunks(chunks)
    index.add_chunks(chunks)
    assert index.chunk_count(V1) == 6


def test_all_chunks_and_embeddings(index):
    chunks = chunk_transcript(_lines("first"), 30, V1)
    index.add_chunks(chunks)
    got = index.all_chunks([V1])
    assert [c.index for c in got] == [0, 1, 2, 3, 4, 5]
    assert got == sorted(chunks, key=lambda c: c.index)
    cs, mat = index.all_embeddings([V1])
    assert mat.shape == (6, index.embedder.dim) and len(cs) == 6
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
    assert index.all_chunks([]) == []


def test_reopening_reuses_collection_and_guards_model(settings):
    cfg = IndexConfig(embedding_model="hash")
    ChromaIndex(settings.chroma_dir, cfg, HashEmbedder()).add_chunks(
        chunk_transcript(_lines("x"), 30, V1)
    )
    again = ChromaIndex(settings.chroma_dir, cfg, HashEmbedder())
    assert len(again) == 6
    with pytest.raises(RuntimeError):
        ChromaIndex(settings.chroma_dir, cfg, HashEmbedder(dim=16))


def test_different_configs_are_isolated(settings):
    a = ChromaIndex(settings.chroma_dir, IndexConfig(embedding_model="hash", child_secs=30), HashEmbedder())
    b = ChromaIndex(settings.chroma_dir, IndexConfig(embedding_model="hash", child_secs=15), HashEmbedder())
    a.add_chunks(chunk_transcript(_lines("x"), 30, V1))
    assert len(a) == 6 and len(b) == 0
