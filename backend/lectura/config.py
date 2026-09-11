"""Typed configuration for the pipeline.

The important split is between *index identity* (anything that changes the
vectors or the chunk boundaries and therefore requires a rebuilt collection)
and *query-time* parameters (which can vary per request without touching the
index). ``IndexConfig`` is the former; ``RetrievalConfig`` and
``GenerationConfig`` are the latter. ``Settings`` bundles them with paths and
server options and knows how to read itself from environment variables or a
plain dict (used by the evaluation harness for YAML configs).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ALLOWED_WHISPER_MODELS = frozenset(
    {
        "tiny.en", "tiny", "base.en", "base", "small.en", "small",
        "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large", "turbo",
    }
)

Space = Literal["cosine", "l2", "ip"]
Strategy = Literal["dense", "bm25", "hybrid"]
ContextOrder = Literal["time", "rank"]
Device = Literal["auto", "cuda", "cpu"]

T = TypeVar("T")


@dataclass(frozen=True)
class IndexConfig:
    """Everything that determines what is stored in a vector collection."""

    child_secs: int = 30
    # bge-small beat MiniLM-L6-v2 by +0.080 MRR@10 (paired bootstrap, CI
    # [+0.040, +0.121], n=283) at the same 384 dimensions; see eval/RESULTS.md.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    doc_prefix: str = ""
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    space: Space = "cosine"
    # HNSW search breadth. Query-time tunable in Chroma, so it is deliberately
    # excluded from the collection identity hash.
    ef_search: int = 100

    def __post_init__(self) -> None:
        if self.child_secs <= 0:
            raise ValueError("child_secs must be positive")
        if self.space not in ("cosine", "l2", "ip"):
            raise ValueError(f"unsupported space: {self.space}")

    def identity(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("ef_search")
        return d

    def collection_name(self) -> str:
        """Deterministic Chroma collection name (3-63 chars, [a-zA-Z0-9._-]).

        Two configs that would produce different vectors or boundaries get
        different names, so ablations never clobber each other or the app.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_model.split("/")[-1].lower()).strip("-")
        digest = hashlib.sha256(
            json.dumps(self.identity(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        name = f"lectura-{slug[:24]}-c{self.child_secs}-{self.space}-{digest}"
        return name[:63].rstrip("-.")


@dataclass(frozen=True)
class RetrievalConfig:
    """How candidates are retrieved and trimmed for a query."""

    strategy: Strategy = "dense"
    candidate_k: int = 20
    rrf_k: int = 60
    # Largest single measured win: +0.150 MRR@10 on the synthetic set (CI
    # [+0.107, +0.195]), and it survives on questions BM25 cannot solve
    # (+0.074, CI [+0.015, +0.132]), so it is not a lexical artefact. Costs
    # ~25 ms per query.
    reranker: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 4
    top_k_playlist: int = 8
    # Minimum score of the best hit for the query to count as "covered".
    # None disables the gate. Scores are 1 - distance for cosine.
    min_score: float | None = None
    parent_secs: int = 120

    def __post_init__(self) -> None:
        if self.strategy not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unsupported strategy: {self.strategy}")
        if self.candidate_k <= 0 or self.top_k <= 0 or self.top_k_playlist <= 0:
            raise ValueError("candidate_k, top_k and top_k_playlist must be positive")
        if self.parent_secs <= 0:
            raise ValueError("parent_secs must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    # None = provider default (2.5 models think by default); 0 disables thinking.
    thinking_budget: int | None = None
    context_order: ContextOrder = "time"
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.context_order not in ("time", "rank"):
            raise ValueError(f"unsupported context_order: {self.context_order}")


_DEFAULT_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:3000", "http://127.0.0.1:3000",
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    whisper_model: str = "tiny.en"
    device: Device = "auto"
    max_playlist_videos: int = 500
    allowed_origins: tuple[str, ...] = _DEFAULT_ORIGINS
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    # Optional server-side fallback for the per-request header. Used by the
    # evaluation harness; the UI still sends its own key.
    gemini_api_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        if self.whisper_model not in ALLOWED_WHISPER_MODELS:
            raise ValueError(f"unsupported whisper model: {self.whisper_model}")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"unsupported device: {self.device}")

    # Derived paths -------------------------------------------------------
    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def videos_file(self) -> Path:
        return self.data_dir / "videos.json"

    @property
    def playlists_file(self) -> Path:
        return self.data_dir / "playlists.json"

    # Construction helpers --------------------------------------------------
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env

        def get(key: str, default: T, cast, strip: bool = True) -> T:
            """Read LECTURA_<key>. ``strip=False`` for prefix values: e5 and bge
            need their trailing space ("query: "), and stripping it would
            silently glue the prefix to the first word of every query."""
            raw = env.get(f"LECTURA_{key}")
            if raw is None or raw.strip() == "":
                return default
            return cast(raw.strip() if strip else raw)

        def opt(cast):
            return lambda raw: None if raw.lower() in ("none", "null", "") else cast(raw)

        index = IndexConfig(
            child_secs=get("CHILD_SECS", IndexConfig.child_secs, int),
            embedding_model=get("EMBEDDING_MODEL", IndexConfig.embedding_model, str),
            doc_prefix=get("DOC_PREFIX", IndexConfig.doc_prefix, str, strip=False),
            query_prefix=get("QUERY_PREFIX", IndexConfig.query_prefix, str, strip=False),
            space=get("SPACE", IndexConfig.space, str),
            ef_search=get("EF_SEARCH", IndexConfig.ef_search, int),
        )
        retrieval = RetrievalConfig(
            strategy=get("STRATEGY", RetrievalConfig.strategy, str),
            candidate_k=get("CANDIDATE_K", RetrievalConfig.candidate_k, int),
            rrf_k=get("RRF_K", RetrievalConfig.rrf_k, int),
            reranker=get("RERANKER", RetrievalConfig.reranker, opt(str)),
            top_k=get("TOP_K", RetrievalConfig.top_k, int),
            top_k_playlist=get("TOP_K_PLAYLIST", RetrievalConfig.top_k_playlist, int),
            min_score=get("MIN_SCORE", RetrievalConfig.min_score, opt(float)),
            parent_secs=get("PARENT_SECS", RetrievalConfig.parent_secs, int),
        )
        generation = GenerationConfig(
            model=get("GEMINI_MODEL", GenerationConfig.model, str),
            temperature=get("TEMPERATURE", GenerationConfig.temperature, float),
            thinking_budget=get("THINKING_BUDGET", GenerationConfig.thinking_budget, opt(int)),
            context_order=get("CONTEXT_ORDER", GenerationConfig.context_order, str),
            max_output_tokens=get("MAX_OUTPUT_TOKENS", GenerationConfig.max_output_tokens, opt(int)),
        )
        origins = get("ALLOWED_ORIGINS", None, lambda s: tuple(o.strip() for o in s.split(",") if o.strip()))
        return cls(
            data_dir=Path(get("DATA_DIR", DEFAULT_DATA_DIR, Path)),
            index=index,
            retrieval=retrieval,
            generation=generation,
            whisper_model=get("WHISPER_MODEL", cls.whisper_model, str),
            device=get("DEVICE", "auto", str),
            max_playlist_videos=get("MAX_PLAYLIST_VIDEOS", cls.max_playlist_videos, int),
            allowed_origins=origins or _DEFAULT_ORIGINS,
            host=get("HOST", "127.0.0.1", str),
            port=get("PORT", 8000, int),
            log_level=get("LOG_LEVEL", "INFO", str),
            gemini_api_key=env.get("GEMINI_API_KEY") or None,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base: Settings | None = None) -> Settings:
        """Overlay a nested dict (e.g. parsed YAML) on ``base`` (default settings).

        Unknown keys raise so a typo in an ablation config cannot silently run
        the baseline.
        """
        base = base or cls()
        data = dict(data)
        sections = {
            "index": (IndexConfig, base.index),
            "retrieval": (RetrievalConfig, base.retrieval),
            "generation": (GenerationConfig, base.generation),
        }
        updates: dict[str, Any] = {}
        for name, (klass, current) in sections.items():
            sub = data.pop(name, None)
            if sub is None:
                continue
            unknown = set(sub) - {f.name for f in fields(klass)}
            if unknown:
                raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
            updates[name] = replace(current, **sub)
        top_level = {f.name for f in fields(cls)} - set(sections)
        unknown = set(data) - top_level
        if unknown:
            raise ValueError(f"unknown settings keys: {sorted(unknown)}")
        if "allowed_origins" in data:
            data["allowed_origins"] = tuple(data["allowed_origins"])
        if "data_dir" in data:
            data["data_dir"] = Path(data["data_dir"])
        return replace(base, **updates, **data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["allowed_origins"] = list(self.allowed_origins)
        d.pop("gemini_api_key", None)  # never serialise secrets
        d["collection_name"] = self.index.collection_name()
        return d
