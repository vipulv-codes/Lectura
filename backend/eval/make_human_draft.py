"""Dump stratified transcript windows so a human can write gold questions.

    python -m eval.make_human_draft --per-video 6 --window 90 --seed 1

Writes eval/data/human_windows.md - regenerated on demand, not committed - with
N windows per corpus video spread over its length, each printed as timestamped
lines. The labeller reads a
window, writes a question + short answer, and records the evidence line
timestamps; `python -m eval.make_human_draft compile draft.tsv` is not provided
on purpose: the draft is written as JSONL directly (see README for the schema)
so nothing is lost in translation.
"""

from __future__ import annotations

import argparse
import random

from lectura.chunking import format_lines, format_ts, parent_window
from lectura.logging_setup import configure_logging
from lectura.transcripts import TranscriptStore

from .common import CORPUS_DATA_DIR, DATA_DIR, load_resolved_corpus


def main() -> None:
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-video", type=int, default=6)
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(DATA_DIR / "human_windows.md"))
    args = ap.parse_args()

    store = TranscriptStore(CORPUS_DATA_DIR / "transcripts")
    rng = random.Random(args.seed)
    lines_out = ["# Transcript windows for human labelling", "",
                 f"seed={args.seed} per_video={args.per_video} window={args.window}s", ""]
    for v in load_resolved_corpus():
        t = store.load(v["video_id"])
        if t is None:
            continue
        dur = t.duration
        lo, hi = 90.0, max(120.0, dur - 90.0)
        step = (hi - lo) / args.per_video
        lines_out.append(f"## {v['video_id']} | {v['course']} | {v['title']} ({format_ts(dur)})")
        for i in range(args.per_video):
            center = lo + (i + 0.5) * step + rng.uniform(-0.25, 0.25) * step
            s, e, lines = parent_window(t.lines, center, args.window)
            lines_out.append(f"\n### window {i + 1}: [{format_ts(s)}] - [{format_ts(e)}]\n")
            lines_out.append("```")
            lines_out.append(format_lines(lines))
            lines_out.append("```")
        lines_out.append("")
    out = args.out
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
