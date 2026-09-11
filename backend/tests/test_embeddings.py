from __future__ import annotations

import numpy as np
import pytest

from lectura.embeddings import Embedder, HashEmbedder, get_embedder, l2_normalize


def test_l2_normalize_rows_and_zero_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    n = l2_normalize(x)
    assert np.allclose(n[0], [0.6, 0.8]) and np.allclose(n[1], [0.0, 0.0])
    assert n.dtype == np.float32
    assert l2_normalize(np.array([3.0, 4.0])).shape == (1, 2)


def test_hash_embedder_is_deterministic_and_unit_norm():
    e = HashEmbedder(dim=128)
    assert isinstance(e, Embedder)
    a = e.embed_documents(["gradient descent", "gradient descent"])
    assert a.shape == (2, 128) and np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert e.embed_documents([]).shape == (0, 128)
    assert e.embed_queries(["x"]).shape == (1, 128)


def test_hash_embedder_similarity_tracks_word_overlap():
    e = HashEmbedder()
    q, near, far = e.embed_queries(["stochastic gradient descent"])[0], *e.embed_documents(
        ["we use stochastic gradient descent here", "the capital of france is paris"]
    )
    assert float(q @ near) > float(q @ far)


def test_empty_text_gets_a_vector():
    e = HashEmbedder()
    assert np.linalg.norm(e.embed_documents([""])[0]) == pytest.approx(1.0)


def test_get_embedder_caches():
    assert get_embedder("hash") is get_embedder("hash")


@pytest.mark.slow
def test_real_minilm_roundtrip():
    e = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    assert e.dim == 384
    vecs = e.embed_documents(["a cat sat on the mat", "stochastic gradient descent"])
    q = e.embed_queries(["optimisation with SGD"])[0]
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert float(q @ vecs[1]) > float(q @ vecs[0])
