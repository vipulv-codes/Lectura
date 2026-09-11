"""Pure functions over transcripts: lines, timestamps, child chunks, parent windows.

Nothing here touches disk, models, or the network, so all of it is unit-tested
exhaustively and shared verbatim by the app and the evaluation harness.

Chunking model
--------------
A transcript is a sorted list of ``Line`` (text, start, duration). Child chunks
are fixed ``child_secs`` windows on a grid starting at 0; each line belongs to
exactly one window: the one containing its *start*. That makes chunk boundaries
deterministic and chunking-independent gold labels (timestamp spans) easy to
score. The window grid (``start``/``end``) and the real extent of the speech in
it (``snippet_start``/``snippet_end``) are both kept, because citations should
point at real speech, not at the grid.

Parent context is *not* stored in the index. It is assembled at query time from
the cached transcript with ``parent_window`` so the parent size is a query-time
knob rather than an index-rebuild.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Line:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "duration": self.duration}

    @classmethod
    def from_dict(cls, d: Any) -> Line:
        """Accepts dicts and any object exposing text/start/duration attributes
        (e.g. youtube-transcript-api snippets)."""
        if isinstance(d, dict):
            return cls(str(d.get("text", "")), float(d.get("start", 0.0)), float(d.get("duration", 0.0)))
        return cls(str(getattr(d, "text", "")), float(getattr(d, "start", 0.0)), float(getattr(d, "duration", 0.0)))


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    video_id: str
    index: int            # grid window index
    start: float          # grid window start (index * child_secs)
    end: float            # grid window end
    snippet_start: float  # start of the first line in the window
    snippet_end: float    # end of the last line in the window (may exceed `end`)
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_lines(items: Iterable[Line]) -> list[Line]:
    """Whitespace-normalise, drop empty lines, clamp negatives, sort by start."""
    out: list[Line] = []
    for ln in items:
        text = normalize_text(ln.text)
        if not text:
            continue
        start = max(0.0, float(ln.start))
        duration = max(0.0, float(ln.duration))
        out.append(Line(text, start, duration))
    out.sort(key=lambda ln: ln.start)
    return out


def transcript_duration(lines: Sequence[Line]) -> float:
    return max((ln.end for ln in lines), default=0.0)


# Timestamps -----------------------------------------------------------------

def format_ts(seconds: float) -> str:
    """``MM:SS`` under an hour, ``HH:MM:SS`` otherwise (no brackets)."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def ts_label(seconds: float) -> str:
    return f"[{format_ts(seconds)}]"


_TS_CORE = r"(\d{1,2}):(\d{2})(?::(\d{2}))?"
_TS_BRACKET_RE = re.compile(r"\[\s*" + _TS_CORE + r"\s*\]")
_TS_BARE_RE = re.compile(r"^\[?\s*" + _TS_CORE + r"\s*\]?$")


def parse_ts(label: str) -> float:
    """``"12:34"``, ``"[12:34]"``, ``"01:02:03"`` -> seconds. Raises ValueError."""
    m = _TS_BARE_RE.match(label.strip())
    if not m:
        raise ValueError(f"not a timestamp: {label!r}")
    a, b, c = m.groups()
    if c is None:
        return int(a) * 60 + int(b)
    return int(a) * 3600 + int(b) * 60 + int(c)


def find_timestamps(text: str) -> list[tuple[str, float]]:
    """All bracketed timestamps in ``text`` in order of appearance, as
    (normalised label without brackets, seconds)."""
    found: list[tuple[str, float]] = []
    for m in _TS_BRACKET_RE.finditer(text):
        secs = parse_ts(m.group(0))
        found.append((format_ts(secs), secs))
    return found


# Chunking -------------------------------------------------------------------

def chunk_transcript(lines: Iterable[Line], child_secs: int, video_id: str) -> list[Chunk]:
    """Single pass: assign every line to the grid window containing its start."""
    if child_secs <= 0:
        raise ValueError("child_secs must be positive")
    groups: dict[int, list[Line]] = {}
    for ln in normalize_lines(lines):
        idx = int(ln.start // child_secs)
        groups.setdefault(idx, []).append(ln)

    chunks: list[Chunk] = []
    for idx in sorted(groups):
        group = groups[idx]
        chunks.append(
            Chunk(
                chunk_id=f"{video_id}:{idx}",
                video_id=video_id,
                index=idx,
                start=float(idx * child_secs),
                end=float((idx + 1) * child_secs),
                snippet_start=group[0].start,
                snippet_end=max(ln.end for ln in group),
                text=" ".join(ln.text for ln in group),
            )
        )
    return chunks


def lines_in_range(lines: Sequence[Line], start: float, end: float) -> list[Line]:
    """Lines whose start lies in the half-open interval [start, end).

    ``lines`` must be sorted by start (``normalize_lines`` guarantees this).
    """
    starts = [ln.start for ln in lines]
    lo = bisect_left(starts, start)
    hi = bisect_left(starts, end)
    return list(lines[lo:hi])


def parent_window(
    lines: Sequence[Line], center: float, parent_secs: int
) -> tuple[float, float, list[Line]]:
    """A ``parent_secs`` window centred on ``center`` (clamped at 0).

    Returns (window_start, window_end, lines). Used to build LLM context around
    a matched child chunk at query time.
    """
    if parent_secs <= 0:
        raise ValueError("parent_secs must be positive")
    p_start = max(0.0, center - parent_secs / 2.0)
    p_end = p_start + parent_secs
    return p_start, p_end, lines_in_range(lines, p_start, p_end)


def format_lines(lines: Iterable[Line]) -> str:
    """One ``[MM:SS] text`` per line; the format the LLM is asked to cite from."""
    return "\n".join(f"{ts_label(ln.start)} {ln.text}" for ln in lines)
