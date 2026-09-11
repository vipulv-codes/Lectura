"""Embedders. Vectors are produced here, not inside ChromaDB, so that

* documents and queries can get different prefixes (e5 / bge style),
* the model can vary per collection for ablations, and
* tests can use a deterministic, download-free embedder.

All embedders return L2-normalised float32 arrays of shape (n, dim).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .devices import resolve_device

log = logging.getLogger(__name__)

HASH_EMBEDDER_NAME = "hash"


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        doc_prefix: str = "",
        query_prefix: str = "",
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # heavy; lazy

        self.name = model_name
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.device = resolve_device(device)
        self.batch_size = batch_size
        log.info("Loading embedding model '%s' on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        # Renamed in sentence-transformers 6; support both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.max_seq_length = int(getattr(self.model, "max_seq_length", 0) or 0)

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return l2_normalize(vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.doc_prefix)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


class HashEmbedder:
    """Deterministic hashed bag-of-words embedder for tests and dry runs.

    No downloads, no torch. Texts sharing words get similar vectors, so
    retrieval round-trips behave sensibly; it is not meant to be good.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.name = HASH_EMBEDDER_NAME
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(tok.encode("utf-8")).digest()  # not security-relevant
            # Two signed buckets per token: a full collision between two
            # different words needs both buckets and both signs to match.
            for a, b, s in ((0, 4, 4), (8, 12, 12)):  # md5 digest is 16 bytes
                bucket = int.from_bytes(digest[a:b], "little") % self.dim
                sign = 1.0 if digest[s] & 1 else -1.0
                v[bucket] += sign
        if not v.any():
            v[0] = 1.0
        return v

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


_EMBEDDERS: dict[tuple, Embedder] = {}
_EMBEDDERS_LOCK = threading.Lock()


def get_embedder(
    model_name: str,
    doc_prefix: str = "",
    query_prefix: str = "",
    device: str = "auto",
    batch_size: int = 64,
) -> Embedder:
    """Process-wide cache so the app and the eval harness share loaded models."""
    key = (model_name, doc_prefix, query_prefix, device, batch_size)
    with _EMBEDDERS_LOCK:
        emb = _EMBEDDERS.get(key)
        if emb is None:
            if model_name == HASH_EMBEDDER_NAME:
                emb = HashEmbedder()
            else:
                emb = SentenceTransformerEmbedder(
                    model_name, doc_prefix, query_prefix, device, batch_size
                )
            _EMBEDDERS[key] = emb
        return emb
