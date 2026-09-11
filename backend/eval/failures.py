"""Failure taxonomy: counts, not adjectives.

    python -m eval.failures --run baseline --mode single
    python -m eval.failures --run G0 --kind generation

Retrieval failures for an answerable query whose gold was not in the top-k:

    ranking_miss    gold is in the candidate pool but below the cutoff
    semantic_miss   gold is not in the candidate pool at all
    wrong_video     nothing from the gold video reached the pool (corpus mode)
    boundary_split  gold span straddles a chunk boundary for this chunk size
                    (reported as an overlay on the two misses above, since it is
                    a property of the labels and the config, not an outcome)

Generation failures:

    no_citation, invalid_citation, mislocalized, false_abstention,
    missed_abstention
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from .common import RESULTS_DIR, read_jsonl, write_json

TOP_K_DEFAULT = 4


def classify_retrieval(rec: dict[str, Any], top_k: int) -> str:
    if not rec.get("answerable", True):
        return "n/a"
    rank = rec.get("first_hit_rank")
    if rank is not None and rank < top_k:
        return "ok"
    if rank is not None:
        return "ranking_miss"
    if rec.get("first_video_rank") is None:
        return "wrong_video"
    return "semantic_miss"


def classify_generation(rec: dict[str, Any]) -> list[str]:
    out = []
    if rec.get("answerable"):
        if rec.get("abstained"):
            out.append("false_abstention")
        else:
            if rec.get("n_citations", 0) == 0:
                out.append("no_citation")
            if rec.get("invalid_citations"):
                out.append("invalid_citation")
            if rec.get("citation_localized") is False:
                out.append("mislocalized")
    elif not rec.get("abstained"):
        out.append("missed_abstention")
    return out or ["ok"]


def run(run_id: str, mode: str, kind: str, top_k: int, examples: int) -> dict[str, Any]:
    path = RESULTS_DIR / run_id / "per_query.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    recs = [r for r in read_jsonl(path) if r.get("mode", mode) == mode]
    counts: Counter[str] = Counter()
    per_rec: list[tuple[str, dict[str, Any]]] = []
    for r in recs:
        if kind == "retrieval":
            label = classify_retrieval(r, top_k)
            counts[label] += 1
            per_rec.append((label, r))
        else:
            for label in classify_generation(r):
                counts[label] += 1
                per_rec.append((label, r))

    total = sum(counts.values())
    print(f"\n== {run_id} ({kind}, mode={mode}, top_k={top_k}) ==")
    for label, n in counts.most_common():
        print(f"  {label:18s} {n:4d}  {100 * n / total:5.1f}%")

    if kind == "retrieval":
        # Straddling is a property of the labels and the chunk grid, so it only
        # means something against its base rate over all answerable queries.
        answerable = [r for label, r in per_rec if label != "n/a"]
        miss_recs = [r for label, r in per_rec if label not in ("ok", "n/a")]
        hit_recs = [r for label, r in per_rec if label == "ok"]
        if miss_recs and answerable:
            base = sum(1 for r in answerable if r.get("straddle")) / len(answerable)
            in_miss = sum(1 for r in miss_recs if r.get("straddle")) / len(miss_recs)
            in_hit = (sum(1 for r in hit_recs if r.get("straddle")) / len(hit_recs)) if hit_recs else float("nan")
            print(
                f"  gold straddles a chunk boundary: {100 * base:.0f}% of all answerable, "
                f"{100 * in_miss:.0f}% of misses, {100 * in_hit:.0f}% of hits"
            )
            out_straddle = {"base_rate": base, "in_misses": in_miss, "in_hits": in_hit}
        else:
            out_straddle = {}
        by_video = Counter(r["video_id"] for label, r in per_rec if label not in ("ok", "n/a"))
        print("  misses by video:", dict(by_video.most_common(5)))
        tag_counts: dict[str, Counter] = {}
        for label, r in per_rec:
            if label in ("ok", "n/a"):
                continue
            for k in ("course", "position", "lexical_tercile"):
                v = (r.get("tags") or {}).get(k)
                if v is not None:
                    tag_counts.setdefault(k, Counter())[str(v)] += 1
        for k, c in tag_counts.items():
            print(f"  misses by {k}:", dict(c.most_common()))

    if examples:
        print("\n  examples:")
        shown: Counter[str] = Counter()
        for label, r in per_rec:
            if label in ("ok", "n/a") or shown[label] >= examples:
                continue
            shown[label] += 1
            q = r.get("question", r.get("qid"))
            print(f"   [{label}] {r['qid']} {q[:90]}")
            if kind == "retrieval" and r.get("top1"):
                print(f"       top1: {r['top1']['video_id']} @{r['top1']['start']:.0f}s score={r['top1']['score']:.3f}  gold anchor={r.get('anchor')}")

    out = {"run_id": run_id, "kind": kind, "mode": mode, "top_k": top_k, "counts": dict(counts), "total": total}
    if kind == "retrieval":
        out["straddle"] = out_straddle
    write_json(RESULTS_DIR / run_id / f"failures_{kind}_{mode}.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="single", choices=["single", "corpus"])
    ap.add_argument("--kind", default="retrieval", choices=["retrieval", "generation"])
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    ap.add_argument("--examples", type=int, default=3, help="failing examples to print per class")
    args = ap.parse_args()
    run(args.run, args.mode, args.kind, args.top_k, args.examples)


if __name__ == "__main__":
    main()
