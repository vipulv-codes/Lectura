"""ChromaDB-backed vector index of child chunks.

One ``ChromaIndex`` wraps one collection whose name is derived from the
``IndexConfig`` (chunk size, embedding model, prefixes, space), so the app and
every ablation config get their own collection inside the same Chroma directory
and never clobber each other.

Chroma is given no embedding function: vectors come from our ``Embedder`` and
are passed explicitly. Only the chunk text and its grid/extent metadata are
stored; parent context is assembled from the transcript at query time.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk
from .config import IndexConfig
from .embeddings import Embedder

log = logging.getLogger(__name__)

_CLIENTS: dict[str, Any] = {}
_CLIENTS_LOCK = threading.Lock()

UPSERT_BATCH = 256


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float   # higher is better; cosine similarity for dense hits
    rank: int      # 0-based rank in the list this hit came from
    source: str = "dense"


def get_client(chroma_dir: Path):
    """One PersistentClient per directory (Chroma refuses conflicting settings
    for the same path, and creating clients is not free)."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    key = str(Path(chroma_dir).resolve())
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            Path(chroma_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=key, settings=ChromaSettings(anonymized_telemetry=False)
            )
            _CLIENTS[key] = client
        return client


def _chunk_from_record(chunk_id: str, document: str, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        video_id=str(meta["video_id"]),
        index=int(meta["index"]),
        start=float(meta["start"]),
        end=float(meta["end"]),
        snippet_start=float(meta.get("snippet_start", meta["start"])),
        snippet_end=float(meta.get("snippet_end", meta["end"])),
        text=document or "",
    )


def _meta_from_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "video_id": chunk.video_id,
        "index": chunk.index,
        "start": chunk.start,
        "end": chunk.end,
        "snippet_start": chunk.snippet_start,
        "snippet_end": chunk.snippet_end,
    }


def _where(video_ids: Sequence[str] | None) -> dict[str, Any] | None:
    if video_ids is None:
        return None
    ids = list(dict.fromkeys(video_ids))
    if len(ids) == 1:
        return {"video_id": ids[0]}
    return {"video_id": {"$in": ids}}


class ChromaIndex:
    def __init__(self, chroma_dir: Path, cfg: IndexConfig, embedder: Embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.client = get_client(chroma_dir)
        self.name = cfg.collection_name()
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            embedding_function=None,
            configuration={"hnsw": {"space": cfg.space, "ef_search": cfg.ef_search}},
            metadata={
                "embedding_model": cfg.embedding_model,
                "dim": embedder.dim,
                "child_secs": cfg.child_secs,
                "doc_prefix": cfg.doc_prefix,
                "query_prefix": cfg.query_prefix,
                "space": cfg.space,
            },
        )
        stored = self.collection.metadata or {}
        if stored.get("embedding_model", cfg.embedding_model) != cfg.embedding_model or int(
            stored.get("dim", embedder.dim)
        ) != int(embedder.dim):
            raise RuntimeError(
                f"collection {self.name} was built with {stored.get('embedding_model')} "
                f"(dim {stored.get('dim')}), not {cfg.embedding_model} (dim {embedder.dim})"
            )

    # Scores ------------------------------------------------------------------

    def _score(self, distance: float) -> float:
        if self.cfg.space == "l2":
            return -float(distance)
        return 1.0 - float(distance)  # cosine and ip both report 1 - similarity

    # Writes ------------------------------------------------------------------

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        total = 0
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors.tolist(),
                documents=[c.text for c in batch],
                metadatas=[_meta_from_chunk(c) for c in batch],
            )
            total += len(batch)
        log.info("Indexed %d chunks into %s", total, self.name)
        return total

    def delete_video(self, video_id: str) -> None:
        self.collection.delete(where={"video_id": video_id})

    def delete_videos(self, video_ids: Sequence[str]) -> None:
        for vid in dict.fromkeys(video_ids):
            self.delete_video(vid)

    # Reads -------------------------------------------------------------------

    def search(self, query: str, video_ids: Sequence[str] | None, k: int) -> list[Hit]:
        """Top-``k`` chunks by similarity, rank order preserved.

        ``video_ids=None`` searches the whole collection; an empty list returns
        nothing (Chroma rejects an empty ``$in``).
        """
        if video_ids is not None and len(video_ids) == 0:
            return []
        if k <= 0:
            return []
        q = self.embedder.embed_queries([query])
        res = self.collection.query(
            query_embeddings=q.tolist(),
            n_results=k,
            where=_where(video_ids),
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0] if res and res.get("ids") else []
        if not ids:
            return []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        hits = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists, strict=True)):
            hits.append(Hit(_chunk_from_record(cid, doc, meta), self._score(dist), rank, "dense"))
        return hits

    def _get_all(self, video_ids: Sequence[str] | None, include: list[str]) -> dict[str, Any]:
        if video_ids is not None and len(video_ids) == 0:
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        return self.collection.get(where=_where(video_ids), include=include)

    def all_chunks(self, video_ids: Sequence[str] | None = None) -> list[Chunk]:
        res = self._get_all(video_ids, ["documents", "metadatas"])
        chunks = [
            _chunk_from_record(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]
        chunks.sort(key=lambda c: (c.video_id, c.index))
        return chunks

    def all_embeddings(
        self, video_ids: Sequence[str] | None = None
    ) -> tuple[list[Chunk], np.ndarray]:
        """Chunks and their stored vectors (for exact-search checks)."""
        res = self._get_all(video_ids, ["documents", "metadatas", "embeddings"])
        chunks = [
            _chunk_from_record(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]
        embs = res.get("embeddings")
        matrix = (
            np.asarray(embs, dtype=np.float32)
            if embs is not None and len(embs)
            else np.zeros((0, self.embedder.dim), dtype=np.float32)
        )
        return chunks, matrix

    def video_ids(self) -> set[str]:
        res = self.collection.get(include=["metadatas"])
        return {str(m["video_id"]) for m in (res.get("metadatas") or []) if m and "video_id" in m}

    def chunk_count(self, video_id: str | None = None) -> int:
        if video_id is None:
            return int(self.collection.count())
        res = self.collection.get(where={"video_id": video_id}, include=[])
        return len(res.get("ids") or [])

    def __len__(self) -> int:
        return self.chunk_count()
