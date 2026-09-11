"""Interactive span-snapping labeller for the gold sets.

    python -m eval.label                 # walk every unverified human item
    python -m eval.label --qid h_001     # jump to one item
    python -m eval.label --set unanswerable
    python -m eval.label --all           # include already-verified items

You type the evidence timestamps exactly as they appear in the transcript
("02:52", or "02:52 02:58" for a range); the tool snaps them to real caption
line boundaries, so a span always runs from the START of the first evidence
line to the END of the last one - the convention make_synthetic.py also uses.
The file is written after every item, so you can stop and resume any time.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from lectura.chunking import Line, format_ts, parse_ts
from lectura.transcripts import TranscriptStore

from .common import CORPUS_DATA_DIR, GOLD_FILES, load_resolved_corpus
from .gold import GoldItem, Span, load_jsonl, save_jsonl

HELP = """
  <ts>            one evidence line, e.g.   02:58
  <ts> <ts>       first and last line, e.g. 02:52 02:58
  ..., ...        several spans:            02:52 02:58, 10:20 10:31
  Enter           keep the current span, mark verified
  ans <text>      replace the short answer
  qn <text>       replace the question
  w               widen the transcript context shown
  s               skip (leave unverified)
  d               delete this item
  x               save and exit
  ?               this help
"""


def snap(lines: list[Line], seconds: float) -> Line:
    """The caption line a typed timestamp refers to.

    You read labels off the screen, so a line whose displayed [MM:SS] label
    matches what you typed wins outright. That matters at boundaries: line
    [02:58] really starts at 178.02, so a typed 178 also falls inside line
    [02:52] (172.50-178.02), and preferring the container would silently pick
    the wrong line. Falls back to the containing line, then the nearest start.
    """
    label = format_ts(seconds)
    exact = [ln for ln in lines if format_ts(ln.start) == label]
    if exact:
        return exact[0]
    inside = [ln for ln in lines if ln.start <= seconds < ln.end]
    if inside:
        return inside[0]
    return min(lines, key=lambda ln: abs(ln.start - seconds))


def parse_spans(text: str, lines: list[Line]) -> list[Span]:
    """Parse '02:52', '02:52 02:58', '02:52-02:58', or a comma-separated list,
    snapping each end to a real caption line."""
    spans: list[Span] = []
    for part in text.split(","):
        cleaned = part.strip().replace("[", "").replace("]", "").replace("-", " ")
        stamps = [p for p in cleaned.split() if p and p.lower() != "to"]
        if not stamps:
            continue
        first = snap(lines, parse_ts(stamps[0]))
        last = snap(lines, parse_ts(stamps[-1])) if len(stamps) > 1 else first
        if last.start < first.start:
            first, last = last, first
        spans.append(Span(round(first.start, 2), round(last.end, 2)))
    return spans


def show_context(lines: list[Line], spans: list[Span], window: float) -> None:
    if spans:
        lo = min(s.start for s in spans) - window
        hi = max(s.end for s in spans) + window
    else:
        lo, hi = 0.0, window * 2
    for ln in lines:
        if not (lo <= ln.start <= hi):
            continue
        marked = any(s.start <= ln.start < s.end or s.start < ln.end <= s.end for s in spans)
        print(f"  {'>' if marked else ' '} [{format_ts(ln.start)}] {ln.start:8.2f}-{ln.end:<8.2f} {ln.text}")


def position_of(anchor: float, duration: float) -> str:
    return "early" if anchor < duration / 3 else "mid" if anchor < 2 * duration / 3 else "late"


def describe(spans: list[Span]) -> str:
    if not spans:
        return "none"
    return ", ".join(
        f"[{format_ts(s.start)}]-[{format_ts(s.end)}] ({s.start:.2f}-{s.end:.2f}, {s.length:.1f}s)" for s in spans
    )


def save(path: Path, items: list[Any]) -> None:
    save_jsonl(path, [i for i in items if i is not None])


def label_answerable(
    items: list[Any], path: Path, store: TranscriptStore,
    titles: dict[str, str], durations: dict[str, float], todo: list[int],
) -> None:
    for n, idx in enumerate(todo, 1):
        it: GoldItem = items[idx]
        transcript = store.load(it.video_id)
        if transcript is None:
            print(f"!! no cached transcript for {it.video_id}; skipping {it.qid}")
            continue
        lines = transcript.lines
        window = 45.0
        while True:
            print("\n" + "=" * 100)
            print(f"[{n}/{len(todo)}] {it.qid}   {it.video_id}   {titles.get(it.video_id, '')[:60]}")
            print(f"  Q: {it.question}")
            print(f"  A: {it.short_answer}")
            print(f"  span: {describe(list(it.spans))}\n")
            show_context(lines, list(it.spans), window)
            try:
                raw = input("\n> ").strip()
            except EOFError:
                save(path, items)
                return
            if raw == "?":
                print(HELP)
                continue
            if raw == "w":
                window += 45.0
                continue
            if raw == "x":
                save(path, items)
                print(f"saved {path.name}")
                return
            if raw == "s":
                break
            if raw == "d":
                items[idx] = None
                save(path, items)
                print("  deleted.")
                break
            if raw.startswith("ans "):
                it = replace(it, short_answer=raw[4:].strip())
                items[idx] = it
                continue
            if raw.startswith("qn "):
                it = replace(it, question=raw[3:].strip())
                items[idx] = it
                continue
            if raw:
                try:
                    new = parse_spans(raw, lines)
                except ValueError as e:
                    print(f"  !! {e}")
                    continue
                if not new:
                    print("  !! no timestamp recognised")
                    continue
                it = replace(it, spans=tuple(new), anchor=min(s.start for s in new))
                for s in new:
                    print(f"  snapped -> [{format_ts(s.start)}]-[{format_ts(s.end)}]  {s.start:.2f}-{s.end:.2f}  ({s.length:.1f}s)")
            if not it.spans:
                print("  !! an answerable item needs at least one span")
                continue
            anchor = min(s.start for s in it.spans)
            it = replace(
                it, anchor=anchor,
                tags={**it.tags, "verified": True, "position": position_of(anchor, durations.get(it.video_id, 1.0))},
            )
            items[idx] = it
            save(path, items)
            print("  verified.")
            break
    save(path, items)


def label_unanswerable(items: list[Any], path: Path, store: TranscriptStore, titles: dict[str, str], todo: list[int]) -> None:
    print("\nConfirm the assigned video does NOT answer the question. '/word' greps its transcript.")
    for n, idx in enumerate(todo, 1):
        it: GoldItem = items[idx]
        transcript = store.load(it.video_id)
        while True:
            print("\n" + "=" * 100)
            print(f"[{n}/{len(todo)}] {it.qid}   {it.video_id}   {titles.get(it.video_id, '')[:60]}")
            print(f"  Q: {it.question}")
            try:
                raw = input("\n  y=not covered (verify)   d=delete (it IS covered)   /word   s=skip   x=exit\n> ").strip()
            except EOFError:
                save(path, items)
                return
            if raw.startswith("/") and transcript is not None:
                term = raw[1:].strip().lower()
                hits = [ln for ln in transcript.lines if term in ln.text.lower()]
                print(f"  {len(hits)} line(s) contain {term!r}")
                for ln in hits[:12]:
                    print(f"    [{format_ts(ln.start)}] {ln.text}")
                continue
            if raw == "x":
                save(path, items)
                print(f"saved {path.name}")
                return
            if raw == "s":
                break
            if raw == "d":
                items[idx] = None
                save(path, items)
                print("  deleted.")
                break
            if raw == "y":
                items[idx] = replace(it, tags={**it.tags, "verified": True})
                save(path, items)
                print("  verified.")
                break
    save(path, items)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="human", choices=["human", "unanswerable"])
    ap.add_argument("--qid", help="label just this item")
    ap.add_argument("--all", action="store_true", help="include already-verified items")
    args = ap.parse_args()

    path = GOLD_FILES[args.set]
    items: list[Any] = load_jsonl(path)
    store = TranscriptStore(CORPUS_DATA_DIR / "transcripts")
    corpus = load_resolved_corpus()
    titles = {v["video_id"]: v["title"] for v in corpus}
    durations = {v["video_id"]: float(v["duration"]) for v in corpus}

    todo = [
        i for i, it in enumerate(items)
        if (args.qid is None or it.qid == args.qid) and (args.all or not it.tags.get("verified", False))
    ]
    if not todo:
        print(f"nothing to do in {path.name} ({len(items)} items)")
        return
    print(f"{len(todo)} item(s) to label in {path.name}.  '?' for help, 'x' to save and exit.")
    if args.set == "human":
        label_answerable(items, path, store, titles, durations, todo)
    else:
        label_unanswerable(items, path, store, titles, todo)
    remaining = sum(1 for it in load_jsonl(path) if not it.tags.get("verified", False))
    print(f"\n{path.name}: {remaining} still unverified.")


if __name__ == "__main__":
    main()
