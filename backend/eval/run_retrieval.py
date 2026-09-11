"""Retrieval evaluation for one config (or the Stage A grid). No LLM calls.

    python -m eval.run_retrieval --config baseline
    python -m eval.run_retrieval --grid stage_a          # 9 indexes x 6 pipelines
    python -m eval.run_retrieval --config baseline --sets human --modes corpus

Per query we retrieve a candidate pool (``--candidate-k``, default 20) once and
derive every cutoff from it. Outputs go to results/<run_id>/:

    metrics.json     aggregate numbers, slices, gate calibration, ANN check
    per_query.jsonl  one record per (query, mode)
    manifest.json    config, git sha, package versions, dataset hashes, device
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from lectura.config import Settings
from lectura.logging_setup import configure_logging

from .common import (
    RESULTS_DIR,
    build_manifest,
    corpus_video_ids,
    dir_size_bytes,
    load_config,
    load_env,
    load_gold,
    make_service,
    settings_from_dict,
    write_json,
    write_jsonl,
)
from .gold import GoldItem, RankedChunk, first_hit_rank, first_video_rank, straddles_boundary
from .metrics import (
    auroc,
    rate_at_or_above,
    slice_records,
    summarize_retrieval,
    threshold_at_tpr,
)

log = logging.getLogger("eval.run_retrieval")

MODES = ("single", "corpus")
SETS = ("human", "synthetic", "unanswerable")

EMBEDDERS = {
    "minilm": {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    "bge-small": {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "e5-small": {"embedding_model": "intfloat/e5-small-v2", "doc_prefix": "passage: ", "query_prefix": "query: "},
}
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ensure_index(svc, video_ids: list[str]) -> dict[str, Any]:
    """Build this config's collection from the cached transcripts if needed."""
    t0 = time.perf_counter()
    built = 0
    for vid in video_ids:
        if svc.index.chunk_count(vid) > 0:
            continue
        if not svc.transcripts.exists(vid):
            raise SystemExit(f"transcript for {vid} not cached; run `python -m eval.build_corpus ingest` first")
        if not svc.catalog.has_video(vid):
            svc.catalog.register_video(vid, title=vid, duration=0, author="", thumbnail="")
        entry = svc.ingest_video(vid) or {}
        if entry.get("status") != "indexed":
            raise SystemExit(f"indexing {vid} failed: {entry.get('error')}")
        built += 1
    dt = time.perf_counter() - t0
    n_chunks = len(svc.index)
    return {
        "collection": svc.index.name,
        "videos_built": built,
        "build_seconds": round(dt, 2),
        "n_chunks": n_chunks,
        "chunks_per_second": round(n_chunks / dt, 1) if built and dt > 0 else None,
        "chroma_dir_bytes": dir_size_bytes(svc.settings.chroma_dir),
        "embedding_dim": svc.embedder.dim,
    }


def to_ranked(hits) -> list[RankedChunk]:
    return [RankedChunk(h.chunk.video_id, h.chunk.snippet_start, h.chunk.snippet_end, h.score, h.chunk.start, h.chunk.end) for h in hits]


def score_query(svc, gold: GoldItem, mode: str, corpus_ids: list[str], candidate_k: int, tol: float) -> dict[str, Any]:
    video_ids = corpus_ids if mode == "corpus" else [gold.video_id]
    t0 = time.perf_counter()
    result = svc.pipeline.retrieve(gold.question, video_ids, k=candidate_k)
    latency = time.perf_counter() - t0
    ranked = to_ranked(result.candidates)
    rec: dict[str, Any] = {
        "qid": gold.qid,
        "set": gold.set,
        "question": gold.question,
        "mode": mode,
        "answerable": gold.answerable,
        "video_id": gold.video_id,
        "anchor": gold.anchor,
        "n_candidates": len(ranked),
        "best_score": result.gate_score,
        "latency_s": round(latency, 5),
        "timings": {k: round(v, 5) for k, v in result.timings.items()},
        "tags": dict(gold.tags),
        "origin_qid": gold.origin_qid,
        "first_hit_rank": None,
        "first_hit_ts": None,
        "ts_error": None,
        "first_video_rank": None,
        "straddle": None,
        "top1": {"video_id": ranked[0].video_id, "start": ranked[0].start, "score": ranked[0].score} if ranked else None,
    }
    if gold.answerable:
        r = first_hit_rank(ranked, gold, tol)
        rec["first_hit_rank"] = r
        rec["first_video_rank"] = first_video_rank(ranked, gold)
        rec["straddle"] = straddles_boundary(gold, svc.settings.index.child_secs)
        if r is not None:
            rec["first_hit_ts"] = ranked[r].start
            rec["ts_error"] = abs(ranked[r].start - (gold.anchor or 0.0))
    return rec


def is_valid_negative(rec: dict[str, Any], mode: str) -> bool:
    """A video-swap negative is a question about video A asked against video B.
    That is unanswerable in single-video mode, but in corpus mode video A is
    back in the search set, so the question IS answerable and the item is not a
    negative at all. Measured: swaps score AUROC 0.52 in corpus mode (chance)
    versus 0.93 in single mode. Counting them in corpus mode understates the
    gate badly, so they are excluded there."""
    if rec["answerable"]:
        return False
    return not (mode == "corpus" and rec.get("tags", {}).get("kind") == "video_swap")


def gate_calibration(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    pos = [r["best_score"] for r in records if r["mode"] == mode and r["answerable"] and r["best_score"] is not None]
    neg = [r["best_score"] for r in records if r["mode"] == mode and is_valid_negative(r, mode) and r["best_score"] is not None]
    if not pos or not neg:
        return {"n_pos": len(pos), "n_neg": len(neg)}
    thr = threshold_at_tpr(pos, 0.95)
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "negatives_excluded_as_answerable_here": sum(
            1 for r in records if r["mode"] == mode and not r["answerable"] and not is_valid_negative(r, mode)
        ),
        "auroc": auroc(pos, neg),
        "threshold_at_95tpr": thr,
        "tpr_at_threshold": rate_at_or_above(pos, thr),
        "fpr_at_threshold": rate_at_or_above(neg, thr),
        "pos_score_median": float(np.median(pos)),
        "neg_score_median": float(np.median(neg)),
    }


def ann_vs_exact(svc, gold: list[GoldItem], corpus_ids: list[str], n: int = 100, k: int = 10, seed: int = 0) -> dict[str, Any] | None:
    """Recall of Chroma's HNSW top-k against brute-force cosine over the stored
    vectors. Only meaningful for the dense stage."""
    if svc.settings.retrieval.strategy != "dense":
        return None
    chunks, matrix = svc.index.all_embeddings(corpus_ids)
    if len(chunks) == 0:
        return None
    rng = random.Random(seed)
    sample = rng.sample([g for g in gold if g.answerable], min(n, len([g for g in gold if g.answerable])))
    if not sample:
        return None
    ids = [c.chunk_id for c in chunks]
    recalls = []
    for g in sample:
        q = svc.embedder.embed_queries([g.question])[0]
        exact = set(np.array(ids)[np.argsort(-(matrix @ q))[:k]].tolist())
        approx = {h.chunk.chunk_id for h in svc.index.search(g.question, corpus_ids, k)}
        recalls.append(len(exact & approx) / k)
    return {"n_queries": len(sample), "k": k, "recall": float(np.mean(recalls)), "n_vectors": len(chunks)}


def run(settings: Settings, run_id: str, sets: list[str], modes: list[str], candidate_k: int, tol: float, seed: int, description: str = "", restrict_available: bool = False, only_videos: list[str] | None = None) -> dict[str, Any]:
    svc = make_service(settings)
    corpus_ids = corpus_video_ids()
    if only_videos:
        # Pin every ASR-degradation run to the same video subset, so the caption
        # reference and the Whisper runs are scored on identical gold items.
        keep_ids = set(only_videos)
        corpus_ids = [v for v in corpus_ids if v in keep_ids]
    if restrict_available:
        # A Whisper data dir only holds the WER subset, so restrict the corpus
        # and gold set to the videos it actually has.
        corpus_ids = [v for v in corpus_ids if svc.transcripts.exists(v)]
    if not corpus_ids:
        raise SystemExit(f"no usable videos under {settings.transcripts_dir}")
    index_info = ensure_index(svc, corpus_ids)
    gold = load_gold(sets)
    if restrict_available or only_videos:
        keep = set(corpus_ids)
        gold = [g for g in gold if g.video_id in keep]
    if not gold:
        raise SystemExit("no gold items loaded")
    log.info("[%s] %d gold items, %d corpus videos, %d chunks", run_id, len(gold), len(corpus_ids), index_info["n_chunks"])

    records: list[dict[str, Any]] = []
    for mode in modes:
        t0 = time.perf_counter()
        for g in gold:
            records.append(score_query(svc, g, mode, corpus_ids, candidate_k, tol))
        log.info("[%s] mode=%s scored %d queries in %.1fs", run_id, mode, len(gold), time.perf_counter() - t0)

    child = settings.index.child_secs
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "description": description,
        "index": index_info,
        "candidate_k": candidate_k,
        "tol": tol,
        "by_mode": {},
        "gate": {},
    }
    for mode in modes:
        mode_recs = [r for r in records if r["mode"] == mode]
        per_set = {}
        for s in sets:
            recs = [r for r in mode_recs if r["set"] == s]
            if recs and any(r["answerable"] for r in recs):
                per_set[s] = summarize_retrieval(recs, child)
        answerable = [r for r in mode_recs if r["answerable"]]
        slices = {}
        for key in ("lexical_tercile", "bm25_easy", "position", "straddle", "course"):
            groups = slice_records(answerable, key)
            if len(groups) > 1:
                slices[key] = {k: summarize_retrieval(v, child) for k, v in sorted(groups.items())}
        metrics["by_mode"][mode] = {"all_answerable": summarize_retrieval(answerable, child), "by_set": per_set, "slices": slices}
        metrics["gate"][mode] = gate_calibration(records, mode)
    if "corpus" in modes:
        metrics["ann_vs_exact"] = ann_vs_exact(svc, gold, corpus_ids, seed=seed)

    out = RESULTS_DIR / run_id
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "per_query.jsonl", records)
    write_json(out / "manifest.json", build_manifest(settings, sets=sets, seed=seed, extra={"kind": "retrieval", "modes": modes}))
    _print_summary(metrics)
    return metrics


def _print_summary(m: dict[str, Any]) -> None:
    print(f"\n== {m['run_id']} ==  chunks={m['index']['n_chunks']}  collection={m['index']['collection']}")
    for mode, block in m["by_mode"].items():
        a = block["all_answerable"]
        print(
            f"  {mode:6s} n={a['n']:4d}  hit@1={a['hit@1']:.3f}  hit@3={a['hit@3']:.3f}  mrr@10={a['mrr@10']:.3f}  "
            f"hit@120s={a['hit@120s']:.3f}  ts_err_med={a['ts_error']['median']:.1f}s  p50={a['latency_s'].get('p50', float('nan')) * 1000:.0f}ms"
        )
        g = m["gate"].get(mode, {})
        if "auroc" in g:
            print(f"         gate auroc={g['auroc']:.3f}  thr@95tpr={g['threshold_at_95tpr']:.3f}  fpr={g['fpr_at_threshold']:.3f}")
    if m.get("ann_vs_exact"):
        print(f"  ann_vs_exact recall@{m['ann_vs_exact']['k']} = {m['ann_vs_exact']['recall']:.3f}")


def stage_a_grid() -> list[tuple[str, dict[str, Any]]]:
    """child x embedder x strategy x reranker; 9 indexes, 54 pipelines."""
    grid = []
    for child, (ename, emb), strategy, rerank in itertools.product((15, 30, 60), EMBEDDERS.items(), ("dense", "bm25", "hybrid"), (False, True)):
        run_id = f"A_c{child}_{ename}_{strategy}{'_rr' if rerank else ''}"
        overlay = {
            "index": {"child_secs": child, **emb},
            "retrieval": {"strategy": strategy, "reranker": RERANKER if rerank else None, "candidate_k": 20},
        }
        grid.append((run_id, overlay))
    return grid


def main() -> None:
    load_env()
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config name under eval/configs or a path")
    ap.add_argument("--grid", choices=["stage_a"], help="run a predefined grid instead of one config")
    ap.add_argument("--run-id", help="override the results directory name")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.0, help="widen gold spans (s), for ASR-degraded runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="substring filter on grid run ids")
    ap.add_argument("--data-dir", help="alternative isolated data dir (e.g. one holding Whisper transcripts)")
    ap.add_argument("--restrict-available", action="store_true", help="limit corpus and gold to videos with a cached transcript in the data dir")
    ap.add_argument("--videos", nargs="*", default=None, help="limit corpus and gold to these video ids")
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.grid:
        for run_id, overlay in stage_a_grid():
            if args.only and args.only not in run_id:
                continue
            settings = settings_from_dict(overlay, run_id, data_dir)
            try:
                run(settings, run_id, args.sets, args.modes, args.candidate_k, args.tol, args.seed, restrict_available=args.restrict_available, only_videos=args.videos)
            except Exception:
                log.exception("grid run %s failed; continuing", run_id)
        return
    if not args.config:
        ap.error("--config or --grid required")
    name, settings, raw = load_config(args.config)
    if data_dir is not None:
        settings = replace(settings, data_dir=data_dir)
    run(settings, args.run_id or name, args.sets, args.modes, args.candidate_k, args.tol, args.seed, raw.get("description", ""), args.restrict_available, args.videos)


if __name__ == "__main__":
    main()
