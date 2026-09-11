"""Generate the synthetic QA set and the video-swap negatives.

    python -m eval.make_synthetic generate --per-hour 25 --seed 0 [--rpm 60]
    python -m eval.make_synthetic swaps --n 15 --seed 0      # from data/human.jsonl
    python -m eval.make_synthetic stats

Generation protocol (see README):
  1. Sample 120 s windows per video, stratified over time (seeded).
  2. Ask Gemini for ONE question a student who remembers the idea (not the
     wording) would ask, a short answer, and the evidence line timestamps.
  3. Deterministic filters: every evidence label must exist in the window; the
     evidence span must be <= 60 s; the question must not be a near-duplicate
     (MiniLM cosine > 0.9) of an accepted one.
  4. A second, cheap yes/no pass: "does this window answer this question?".
  5. Tags for slicing: lexical overlap tercile, bm25_easy (BM25 over the video's
     own 30 s chunks ranks a gold chunk first), position (early/mid/late), course.

Every Gemini call is cached, so re-running is free once the set exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from collections import Counter
from typing import Any

import numpy as np

from lectura.chunking import chunk_transcript, format_lines, format_ts, parent_window, parse_ts
from lectura.config import GenerationConfig
from lectura.embeddings import get_embedder
from lectura.generation import GeminiGenerator
from lectura.logging_setup import configure_logging
from lectura.retrieval import tokenize
from lectura.transcripts import TranscriptStore

from .common import (
    CORPUS_DATA_DIR,
    GOLD_FILES,
    CachedGemini,
    load_env,
    load_resolved_corpus,
)
from .gold import SET_SYNTHETIC, SET_UNANSWERABLE, GoldItem, Span, is_hit, load_jsonl, save_jsonl

log = logging.getLogger("eval.make_synthetic")

WINDOW_SECS = 120
MAX_SPAN_SECS = 60.0
DEDUPE_COSINE = 0.9

GEN_SYSTEM = (
    "You write evaluation questions for a lecture question-answering system. You are given a short "
    "window of a lecture transcript where every line is prefixed with its start time in brackets.\n"
    "Write ONE question that a student who watched the lecture and remembers the idea, but not the "
    "lecturer's exact words, would plausibly ask. Requirements:\n"
    "- It must be answerable from THIS window alone, and the answer must be specific (not a yes/no).\n"
    "- Paraphrase: do not reuse distinctive phrases from the transcript in the question.\n"
    "- Do not mention 'the transcript', 'the window', 'the lecturer says' or timestamps in the question.\n"
    "- Give a short answer (one sentence, at most 25 words).\n"
    "- List the timestamps (exactly as they appear in the brackets) of the one to three lines that contain the "
    "evidence for the answer; they must be adjacent or nearly adjacent.\n"
    "Reply with ONLY a JSON object: {\"question\": ..., \"short_answer\": ..., \"evidence_timestamps\": [\"MM:SS\", ...]}.\n"
    "If the window has no self-contained, question-worthy idea (e.g. it is only chit-chat or an "
    "unfinished setup), reply with {\"question\": null}."
)

VERIFY_SYSTEM = (
    "You are checking an evaluation item. Given a transcript window and a question, answer with exactly "
    "'yes' if the window contains enough information to answer the question specifically, otherwise 'no'. "
    "Reply with a single word."
)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(_strip_fences(text))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _content_words(text: str) -> set[str]:
    return set(tokenize(text))


def sample_centers(duration: float, n: int, rng: random.Random) -> list[float]:
    """Evenly spaced centres with jitter, avoiding the first/last minute."""
    lo, hi = min(60.0, duration / 4), max(duration - 60.0, duration * 3 / 4)
    if n <= 0 or hi <= lo:
        return []
    step = (hi - lo) / n
    return [lo + (i + 0.5) * step + rng.uniform(-0.3, 0.3) * step for i in range(n)]


def bm25_easy(video_lines, gold_spans: list[Span], question: str, video_id: str) -> bool:
    """Does plain BM25 over the video's own 30 s chunks rank a gold chunk first?"""
    from rank_bm25 import BM25Okapi

    chunks = chunk_transcript(video_lines, 30, video_id)
    if not chunks:
        return False
    bm = BM25Okapi([tokenize(c.text) or ["<empty>"] for c in chunks])
    toks = tokenize(question)
    if not toks:
        return False
    scores = bm.get_scores(toks)
    top = chunks[int(np.argmax(scores))]
    gold = GoldItem(qid="tmp", set=SET_SYNTHETIC, question=question, video_id=video_id, spans=tuple(gold_spans))
    return is_hit(video_id, top.snippet_start, top.snippet_end, gold)


def generate(per_hour: float, seed: int, rpm: float | None, model: str, limit: int | None) -> None:
    videos = load_resolved_corpus()
    store = TranscriptStore(CORPUS_DATA_DIR / "transcripts")
    gen = CachedGemini(GeminiGenerator(_api_key(), GenerationConfig(model=model, temperature=0.7, thinking_budget=0)), rpm=rpm)
    verifier = CachedGemini(GeminiGenerator(_api_key(), GenerationConfig(model=model, temperature=0.0, thinking_budget=0)), rpm=rpm)
    rng = random.Random(seed)
    reasons: Counter[str] = Counter()
    accepted: list[GoldItem] = []
    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2")
    accepted_vecs: list[np.ndarray] = []

    for v in videos:
        vid = v["video_id"]
        transcript = store.load(vid)
        if transcript is None:
            log.warning("no transcript for %s; run build_corpus ingest", vid)
            continue
        duration = transcript.duration
        n = max(3, math.ceil(duration / 3600 * per_hour))
        centers = sample_centers(duration, n, rng)
        log.info("%s: %d windows over %.0f min", vid, len(centers), duration / 60)
        for w_idx, center in enumerate(centers):
            if limit is not None and len(accepted) >= limit:
                break
            w_start, w_end, lines = parent_window(transcript.lines, center, WINDOW_SECS)
            if len(lines) < 6:
                reasons["window_too_short"] += 1
                continue
            labels = {format_ts(ln.start): ln for ln in lines}
            user = f"Lecture: {v['title']}\nWindow {format_ts(w_start)}-{format_ts(w_end)}\n\n{format_lines(lines)}"
            completion = gen.complete(GEN_SYSTEM, user, tag=f"gen:{vid}:{w_idx}")
            obj = _parse_json(completion.text)
            if not obj or not obj.get("question"):
                reasons["no_question"] += 1
                continue
            question = str(obj["question"]).strip()
            short = str(obj.get("short_answer") or "").strip()
            ev = [str(t).strip("[] ") for t in (obj.get("evidence_timestamps") or [])]
            ev_lines = []
            bad = False
            for t in ev:
                try:
                    key = format_ts(parse_ts(t))
                except ValueError:
                    bad = True
                    break
                if key not in labels:
                    bad = True
                    break
                ev_lines.append(labels[key])
            if bad or not ev_lines:
                reasons["bad_evidence_label"] += 1
                continue
            span = Span(min(ln.start for ln in ev_lines), max(ln.end for ln in ev_lines))
            if span.length > MAX_SPAN_SECS:
                reasons["span_too_long"] += 1
                continue
            if re.search(r"\b(transcript|window|timestamp|lecturer says)\b", question, re.I):
                reasons["meta_question"] += 1
                continue
            # Near-duplicate check
            qvec = embedder.embed_queries([question])[0]
            if accepted_vecs and float(np.max(np.stack(accepted_vecs) @ qvec)) > DEDUPE_COSINE:
                reasons["duplicate"] += 1
                continue
            # Verification pass
            verdict = verifier.complete(VERIFY_SYSTEM, f"Transcript window:\n{format_lines(lines)}\n\nQuestion: {question}", tag=f"verify:{vid}:{w_idx}")
            if not verdict.text.strip().lower().startswith("yes"):
                reasons["verifier_no"] += 1
                continue
            q_words = _content_words(question)
            ev_words = _content_words(" ".join(ln.text for ln in ev_lines))
            overlap = len(q_words & ev_words) / len(q_words) if q_words else 0.0
            position = "early" if center < duration / 3 else "mid" if center < 2 * duration / 3 else "late"
            item = GoldItem(
                qid=f"s_{vid}_{w_idx:03d}",
                set=SET_SYNTHETIC,
                question=question,
                short_answer=short or None,
                video_id=vid,
                spans=(span,),
                tags={
                    "course": v["course"],
                    "position": position,
                    "lexical_overlap": round(overlap, 3),
                    "bm25_easy": bm25_easy(transcript.lines, [span], question, vid),
                    "window": [round(w_start, 1), round(w_end, 1)],
                    "generator": completion.model_version or model,
                },
            )
            accepted.append(item)
            accepted_vecs.append(qvec)
            reasons["accepted"] += 1

    # Lexical terciles over the accepted set
    if accepted:
        overlaps = sorted(it.tags["lexical_overlap"] for it in accepted)
        t1, t2 = overlaps[len(overlaps) // 3], overlaps[2 * len(overlaps) // 3]
        for it in accepted:
            o = it.tags["lexical_overlap"]
            it.tags["lexical_tercile"] = "low" if o < t1 else "mid" if o < t2 else "high"
    save_jsonl(GOLD_FILES["synthetic"], accepted)
    log.info("Synthetic set: %d accepted -> %s", len(accepted), GOLD_FILES["synthetic"])
    log.info("Filter counts: %s", dict(reasons))
    log.info("Gemini cache: gen %s verify %s", gen.stats, verifier.stats)


def swaps(n: int, seed: int, source_set: str) -> None:
    """Video-swap negatives: a human question asked against a different video
    of the same course. Guaranteed unanswerable, in-domain, hard."""
    src = load_jsonl(GOLD_FILES[source_set])
    videos = load_resolved_corpus()
    by_course: dict[str, list[str]] = {}
    for v in videos:
        by_course.setdefault(v["course"], []).append(v["video_id"])
    course_of = {v["video_id"]: v["course"] for v in videos}
    rng = random.Random(seed)
    candidates = [g for g in src if g.answerable and len(by_course.get(course_of.get(g.video_id, ""), [])) > 1]
    rng.shuffle(candidates)
    out: list[GoldItem] = []
    existing = load_jsonl(GOLD_FILES["unanswerable"]) if GOLD_FILES["unanswerable"].exists() else []
    kept = [g for g in existing if g.tags.get("kind") != "video_swap"]
    for g in candidates[:n]:
        others = [x for x in by_course[course_of[g.video_id]] if x != g.video_id]
        target = rng.choice(others)
        out.append(
            GoldItem(
                qid=f"u_swap_{g.qid}",
                set=SET_UNANSWERABLE,
                question=g.question,
                video_id=target,
                origin_qid=g.qid,
                tags={"kind": "video_swap", "course": course_of[target], "origin_video": g.video_id},
            )
        )
    save_jsonl(GOLD_FILES["unanswerable"], kept + out)
    log.info("Wrote %d video-swap negatives (+%d kept hand-written) -> %s", len(out), len(kept), GOLD_FILES["unanswerable"])


def stats() -> None:
    for name, path in GOLD_FILES.items():
        if not path.exists():
            print(f"{name:12s} missing")
            continue
        items = load_jsonl(path)
        by_video = Counter(it.video_id for it in items)
        tags = Counter()
        for it in items:
            for k in ("lexical_tercile", "position", "kind"):
                if k in it.tags:
                    tags[f"{k}={it.tags[k]}"] += 1
        print(f"{name:12s} n={len(items):4d} videos={len(by_video):2d} answerable={sum(it.answerable for it in items)}  {dict(tags)}")


def _api_key() -> str:
    import os

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set (backend/.env)")
    return key


def main() -> None:
    load_env()
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--per-hour", type=float, default=25.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--rpm", type=float, default=None)
    g.add_argument("--model", default="gemini-2.5-flash")
    g.add_argument("--limit", type=int, default=None, help="stop after N accepted (smoke runs)")
    s = sub.add_parser("swaps")
    s.add_argument("--n", type=int, default=15)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--from-set", default="human", choices=["human", "synthetic"])
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.command == "generate":
        generate(args.per_hour, args.seed, args.rpm, args.model, args.limit)
    elif args.command == "swaps":
        swaps(args.n, args.seed, args.from_set)
    else:
        stats()


if __name__ == "__main__":
    main()
