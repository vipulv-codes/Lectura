"""Prompt construction and Gemini calls.

``build_prompt`` is pure (given hits + cached transcripts) and returns, besides
the text, the set of timestamp labels that actually appear in the context. The
evaluation harness uses that set to score citation validity deterministically,
without re-parsing the prompt.

Parent windows of neighbouring hits from the same video are merged so the model
never sees the same line twice and the context-seconds figure is honest.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .chunking import Line, find_timestamps, format_lines, format_ts, parent_window, ts_label
from .config import GenerationConfig
from .index import Hit
from .transcripts import Transcript, TranscriptStore

log = logging.getLogger(__name__)

ABSTAIN_SENTINEL = "Not covered in the retrieved lecture segments."


@dataclass(frozen=True)
class Source:
    timestamp: str
    seconds: float
    video_id: str
    video_title: str
    snippet: str
    score: float
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "seconds": self.seconds,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "snippet": self.snippet,
            "score": self.score,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class ContextBlock:
    video_id: str
    title: str
    start: float
    end: float
    matched: tuple[float, ...]   # snippet starts of the hits inside this block
    best_rank: int
    lines: tuple[Line, ...]

    def render(self) -> str:
        marks = " ".join(ts_label(m) for m in self.matched)
        return (
            f"Lecture: '{self.title}' | matched at {marks} | window {ts_label(self.start)}-{ts_label(self.end)}\n"
            f"{format_lines(self.lines)}"
        )


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    valid_timestamps: frozenset[str]
    context_seconds: float
    context_chars: int
    blocks: tuple[ContextBlock, ...]
    sources: tuple[Source, ...]


def _merge_windows(
    hits: Sequence[Hit], transcript: Transcript, title: str, parent_secs: int
) -> list[ContextBlock]:
    spans = []
    for h in hits:
        s, e, _ = parent_window(transcript.lines, h.chunk.mid, parent_secs)
        spans.append((s, e, h))
    spans.sort(key=lambda t: t[0])
    merged: list[list[Any]] = []
    for s, e, h in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].append(h)
        else:
            merged.append([s, e, [h]])
    blocks = []
    for s, e, hs in merged:
        _, _, lines = parent_window(transcript.lines, (s + e) / 2, round(e - s))
        blocks.append(
            ContextBlock(
                video_id=transcript.video_id,
                title=title,
                start=s,
                end=e,
                matched=tuple(sorted(h.chunk.snippet_start for h in hs)),
                best_rank=min(h.rank for h in hs),
                lines=tuple(lines),
            )
        )
    return blocks


def build_context(
    hits: Sequence[Hit],
    store: TranscriptStore,
    titles: Mapping[str, str],
    parent_secs: int,
    order: str = "time",
) -> list[ContextBlock]:
    by_video: dict[str, list[Hit]] = {}
    for h in hits:
        by_video.setdefault(h.chunk.video_id, []).append(h)
    blocks: list[ContextBlock] = []
    for vid, vhits in by_video.items():
        title = titles.get(vid, f"Video {vid}")
        transcript = store.load(vid)
        if transcript is None:
            # Transcript cache missing (deleted mid-flight): fall back to chunk text.
            for h in vhits:
                blocks.append(
                    ContextBlock(
                        vid, title, h.chunk.start, h.chunk.end, (h.chunk.snippet_start,), h.rank,
                        (Line(h.chunk.text, h.chunk.snippet_start, h.chunk.snippet_end - h.chunk.snippet_start),),
                    )
                )
            continue
        blocks.extend(_merge_windows(vhits, transcript, title, parent_secs))
    if order == "rank":
        blocks.sort(key=lambda b: b.best_rank)
    else:
        # Videos in order of their best hit; chronological inside a video.
        first_rank = {b.video_id: min(x.best_rank for x in blocks if x.video_id == b.video_id) for b in blocks}
        blocks.sort(key=lambda b: (first_rank[b.video_id], b.start))
    return blocks


SYSTEM_TEMPLATE = (
    "You are an expert teaching assistant for a university lecture course. Answer the student's "
    "question using ONLY the lecture transcript segments below.\n\n"
    "Transcript context:\n"
    "-----------------\n"
    "{context}\n"
    "-----------------\n\n"
    "Rules:\n"
    "1. Use only the provided context. If the question cannot be answered from it, reply with exactly "
    "this sentence first: \"{sentinel}\" You may then add one sentence saying what the retrieved "
    "segments do cover.\n"
    "2. Be clear, educational and structured; use markdown.\n"
    "3. CRITICAL: cite the timestamp of the specific transcript line that supports each key point, "
    "in the format [MM:SS] or [HH:MM:SS], copied exactly from the line prefixes above. One timestamp "
    "per bracket pair: write '[12:34] and [15:20]', never '[12:34, 15:20]'. Cite where an explanation "
    "begins, not every consecutive line. When segments come from different lectures, name the "
    "lecture in prose and keep the timestamp in brackets. These become clickable links.\n"
    "4. Never cite a timestamp that does not appear in the context."
)


def build_prompt(
    query: str,
    hits: Sequence[Hit],
    store: TranscriptStore,
    titles: Mapping[str, str],
    parent_secs: int,
    order: str = "time",
) -> Prompt:
    blocks = build_context(hits, store, titles, parent_secs, order)
    context = "\n\n=====================\n\n".join(b.render() for b in blocks)
    valid = {format_ts(ln.start) for b in blocks for ln in b.lines}
    valid |= {format_ts(m) for b in blocks for m in b.matched}
    seconds = sum(b.end - b.start for b in blocks)
    sources = tuple(
        Source(
            timestamp=format_ts(h.chunk.snippet_start),
            seconds=h.chunk.snippet_start,
            video_id=h.chunk.video_id,
            video_title=titles.get(h.chunk.video_id, f"Video {h.chunk.video_id}"),
            snippet=h.chunk.text,
            score=h.score,
            chunk_id=h.chunk.chunk_id,
        )
        for h in hits
    )
    return Prompt(
        system=SYSTEM_TEMPLATE.format(context=context, sentinel=ABSTAIN_SENTINEL),
        user=f"Student question: {query}",
        valid_timestamps=frozenset(valid),
        context_seconds=seconds,
        context_chars=len(context),
        blocks=tuple(blocks),
        sources=sources,
    )


# Gemini ---------------------------------------------------------------------

class GenerationError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    latency_s: float
    model: str
    model_version: str | None


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[Source, ...]
    abstained: bool
    citations: tuple[tuple[str, float], ...]
    invalid_citations: tuple[str, ...]
    usage: Usage
    latency_s: float
    model: str
    model_version: str | None
    context_seconds: float
    prompt_chars: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.text,
            "sources": [s.to_dict() for s in self.sources],
            "abstained": self.abstained,
            "citations": [{"label": lbl, "seconds": secs} for lbl, secs in self.citations],
            "invalid_citations": list(self.invalid_citations),
            "usage": self.usage.to_dict(),
            "latency_s": round(self.latency_s, 3),
            "model": self.model,
            "model_version": self.model_version,
            "context_seconds": self.context_seconds,
            **self.extra,
        }


def _classify_error(e: Exception) -> int:
    text = f"{type(e).__name__}: {e}".lower()
    status = getattr(e, "code", None) or getattr(e, "status_code", None)
    if status in (400, 401, 403) and ("api key" in text or "permission" in text or "api_key" in text):
        return 401
    if status == 429 or "resource_exhausted" in text or "quota" in text or "rate limit" in text:
        return 429
    if "api key" in text or "api_key" in text or "permission_denied" in text or "unauthenticated" in text:
        return 401
    return 502


def parse_answer(text: str, prompt: Prompt) -> tuple[bool, tuple[tuple[str, float], ...], tuple[str, ...]]:
    abstained = text.strip().startswith(ABSTAIN_SENTINEL) or ABSTAIN_SENTINEL in text[:200]
    cites = tuple(find_timestamps(text))
    invalid = tuple(sorted({lbl for lbl, _ in cites if lbl not in prompt.valid_timestamps}))
    return abstained, cites, invalid


class GeminiGenerator:
    def __init__(self, api_key: str, cfg: GenerationConfig) -> None:
        if not api_key:
            raise GenerationError("Missing Gemini API key", status=401)
        self.api_key = api_key
        self.cfg = cfg
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy: keeps import light for tests

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _config(self, system: str):
        from google.genai import types

        kwargs: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "system_instruction": system,
        }
        if self.cfg.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.cfg.max_output_tokens
        if self.cfg.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.cfg.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def complete(self, system: str, user: str) -> Completion:
        client = self._get_client()
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self.cfg.model, contents=[user], config=self._config(system)
            )
        except Exception as e:
            status = _classify_error(e)
            log.warning("Gemini call failed (%s): %s", status, e)
            raise GenerationError(f"{type(e).__name__}: {e}", status=status) from e
        latency = time.perf_counter() - t0
        text = response.text
        if not text:
            raise GenerationError("Gemini returned an empty response (blocked or truncated)", 502)
        um = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=int(getattr(um, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(um, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(um, "thoughts_token_count", 0) or 0),
            total_tokens=int(getattr(um, "total_token_count", 0) or 0),
        )
        return Completion(text, usage, latency, self.cfg.model, getattr(response, "model_version", None))

    def answer(self, prompt: Prompt) -> Answer:
        completion = self.complete(prompt.system, prompt.user)
        abstained, cites, invalid = parse_answer(completion.text, prompt)
        return Answer(
            text=completion.text,
            sources=prompt.sources,
            abstained=abstained,
            citations=cites,
            invalid_citations=invalid,
            usage=completion.usage,
            latency_s=completion.latency_s,
            model=completion.model,
            model_version=completion.model_version,
            context_seconds=prompt.context_seconds,
            prompt_chars=len(prompt.system) + len(prompt.user),
        )

    def chapters(self, transcript: Transcript, title: str) -> Completion:
        prompt = (
            "You are an academic note-taker. Analyse the lecture transcript below and produce a "
            "structured chapter outline.\n\n"
            "Format: a Markdown list of headings and bullet points, no tables (never use | characters). "
            "Structure each chapter exactly like this:\n"
            "### [MM:SS] Chapter Title\n"
            "- Bullet summarising the first concept\n"
            "- Bullet summarising the second concept\n\n"
            "Timestamps must be copied from the transcript line prefixes and must mark real topic "
            "shifts.\n\n"
            f"Lecture title: {title}\n"
            "Lecture transcript:\n"
            "------------------\n"
            f"{format_lines(transcript.lines)}\n"
            "------------------\n"
        )
        return self.complete("You are a high-quality lecture note-taker and summariser.", prompt)


def no_context_answer(query: str) -> Answer:
    """Returned without calling the model when nothing was retrieved."""
    return Answer(
        text=f"{ABSTAIN_SENTINEL} No indexed transcript content matched the question.",
        sources=(),
        abstained=True,
        citations=(),
        invalid_citations=(),
        usage=Usage(),
        latency_s=0.0,
        model="none",
        model_version=None,
        context_seconds=0.0,
        prompt_chars=0,
        extra={"query": query},
    )
