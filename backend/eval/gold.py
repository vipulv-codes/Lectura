"""Gold labels and the relevance rule.

Gold is *chunking-independent*: each item names a video and one or more
timestamp spans of the evidence, never a chunk id, so the same labels score
every chunk-size ablation. A retrieved chunk is a hit when it comes from the
right video and its speech extent overlaps a gold span by at least 5 seconds
(or half the span, if the span is shorter than 10 s). That rejects one-second
boundary grazes and is stable when chunk boundaries move.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SET_HUMAN = "human"
SET_SYNTHETIC = "synthetic"
SET_UNANSWERABLE = "unanswerable"
SETS = (SET_HUMAN, SET_SYNTHETIC, SET_UNANSWERABLE)

MIN_OVERLAP_SECS = 5.0


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"span end {self.end} before start {self.start}")

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class GoldItem:
    qid: str
    set: str
    question: str
    video_id: str
    spans: tuple[Span, ...] = ()
    short_answer: str | None = None
    anchor: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    origin_qid: str | None = None

    def __post_init__(self) -> None:
        if self.set not in SETS:
            raise ValueError(f"unknown set {self.set!r}")
        if self.anchor is None and self.spans:
            object.__setattr__(self, "anchor", min(s.start for s in self.spans))

    @property
    def answerable(self) -> bool:
        return bool(self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "set": self.set,
            "question": self.question,
            "short_answer": self.short_answer,
            "video_id": self.video_id,
            "spans": [{"start": s.start, "end": s.end} for s in self.spans],
            "anchor": self.anchor,
            "tags": dict(self.tags),
            "origin_qid": self.origin_qid,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldItem:
        return cls(
            qid=str(d["qid"]),
            set=str(d["set"]),
            question=str(d["question"]),
            video_id=str(d["video_id"]),
            spans=tuple(Span(float(s["start"]), float(s["end"])) for s in d.get("spans") or ()),
            short_answer=d.get("short_answer"),
            anchor=None if d.get("anchor") is None else float(d["anchor"]),
            tags=dict(d.get("tags") or {}),
            origin_qid=d.get("origin_qid"),
        )


def load_jsonl(path: Path) -> list[GoldItem]:
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(GoldItem.from_dict(json.loads(line)))
    return items


def save_jsonl(path: Path, items: Iterable[GoldItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Relevance -------------------------------------------------------------------

def span_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def is_hit(
    chunk_video: str, chunk_start: float, chunk_end: float, gold: GoldItem, tol: float = 0.0
) -> bool:
    """Does a chunk with speech extent [chunk_start, chunk_end] contain gold
    evidence? ``tol`` widens the gold spans (used when scoring against ASR
    transcripts whose timestamps drift relative to the caption cues)."""
    if chunk_video != gold.video_id:
        return False
    for s in gold.spans:
        overlap = span_overlap(chunk_start, chunk_end, s.start - tol, s.end + tol)
        if overlap >= min(MIN_OVERLAP_SECS, 0.5 * (s.length + 2 * tol)) and overlap > 0:
            return True
    return False


@dataclass(frozen=True)
class RankedChunk:
    """The subset of a retrieval hit that scoring needs."""

    video_id: str
    start: float          # speech extent start (chunk.snippet_start)
    end: float            # speech extent end (chunk.snippet_end)
    score: float
    grid_start: float = 0.0
    grid_end: float = 0.0


def first_hit_rank(ranked: Sequence[RankedChunk], gold: GoldItem, tol: float = 0.0) -> int | None:
    for rank, c in enumerate(ranked):
        if is_hit(c.video_id, c.start, c.end, gold, tol):
            return rank
    return None


def first_video_rank(ranked: Sequence[RankedChunk], gold: GoldItem) -> int | None:
    """Rank of the first chunk from the gold video (video-level routing)."""
    seen: list[str] = []
    for c in ranked:
        if c.video_id not in seen:
            seen.append(c.video_id)
        if c.video_id == gold.video_id:
            return len(seen) - 1
    return None


def straddles_boundary(gold: GoldItem, child_secs: int) -> bool:
    """Does any gold span cross a child-window boundary for this chunk size?
    Explains many near-misses in small-window configs."""
    for s in gold.spans:
        if int(s.start // child_secs) != int(max(s.start, s.end - 1e-6) // child_secs):
            return True
    return False
