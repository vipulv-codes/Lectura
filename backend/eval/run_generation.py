"""Generation evaluation for one config: answers, citations, abstention, cost,
and (optionally) an LLM judge for faithfulness and correctness.

    python -m eval.run_generation --config baseline --run-id G0
    python -m eval.run_generation --config baseline --run-id G0 --judge gemini-3-flash-preview

A --judge run also writes results/<run>/judge_sample.jsonl; grade it against a
person with `python -m eval.grade --run <run>`.

Query set (default): all human items, all unanswerable items, and a seeded
subset of synthetic items (``--synthetic-n``). Single-video mode by default,
which is the product's main use; ``--modes corpus`` adds playlist-style runs.

Deterministic metrics need no judge: citation validity (against the exact set
of timestamps that were in the prompt), citation localisation (first citation
inside the gold span ± 10 s), abstention by sentinel, tokens, latency, cost.
Judge metrics are reported separately and only as strong as the human
agreement measured on ``judge_sample.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from statistics import mean
from typing import Any

from lectura.config import GenerationConfig, Settings
from lectura.generation import Answer, GeminiGenerator, Prompt, parse_answer
from lectura.logging_setup import configure_logging

from .common import (
    RESULTS_DIR,
    CachedGemini,
    build_manifest,
    corpus_video_ids,
    load_config,
    load_env,
    load_gold,
    load_pricing,
    make_service,
    price_for,
    write_json,
    write_jsonl,
)
from .gold import GoldItem
from .metrics import (
    citation_localized,
    citation_validity,
    estimate_cost_usd,
    percentile,
)
from .run_retrieval import ensure_index

log = logging.getLogger("eval.run_generation")

JUDGE_FAITHFUL_SYSTEM = (
    "You are grading whether an answer is supported by the provided lecture transcript context. "
    "Consider every factual claim in the answer. Reply with ONLY a JSON object: "
    '{"verdict": "supported" | "partially_supported" | "unsupported", "unsupported_claims": ["..."]}. '
    "'supported' means every claim is stated or directly implied by the context; 'partially_supported' means "
    "the main point is supported but some detail is not; 'unsupported' means the main point is not in the context. "
    "Ignore style. Do not use outside knowledge to decide support."
)

JUDGE_CORRECT_SYSTEM = (
    "You are grading an answer against a reference short answer. Reply with ONLY a JSON object: "
    '{"verdict": "correct" | "partially_correct" | "incorrect"}. '
    "'correct' means the answer conveys the reference's meaning (extra correct detail is fine); "
    "'partially_correct' means it gets part of it; 'incorrect' means it contradicts or misses it. "
    "If the answer says the topic is not covered, grade it 'incorrect'."
)


class CachedAnswerer:
    """Drop-in for GeminiGenerator.answer that routes through the disk cache."""

    def __init__(self, cached: CachedGemini) -> None:
        self.cached = cached

    def answer(self, prompt: Prompt) -> Answer:
        c = self.cached.complete(prompt.system, prompt.user, tag="answer")
        abstained, cites, invalid = parse_answer(c.text, prompt)
        return Answer(
            text=c.text, sources=prompt.sources, abstained=abstained, citations=cites, invalid_citations=invalid,
            usage=c.usage, latency_s=c.latency_s, model=c.model, model_version=c.model_version,
            context_seconds=prompt.context_seconds, prompt_chars=len(prompt.system) + len(prompt.user),
        )


class _CapturingAnswerer:
    """Records the prompt the service built so the judge can see the context."""

    def __init__(self, inner: CachedAnswerer) -> None:
        self.inner = inner
        self.prompt: Prompt | None = None

    def answer(self, prompt: Prompt) -> Answer:
        self.prompt = prompt
        return self.inner.answer(prompt)


def _parse_verdict(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("verdict", "unparsed"))
        except json.JSONDecodeError:
            pass
    return "unparsed"


def select_queries(sets: list[str], synthetic_n: int | None, seed: int) -> list[GoldItem]:
    items = load_gold([s for s in sets if s != "synthetic"])
    if "synthetic" in sets:
        syn = load_gold(["synthetic"])
        if synthetic_n is not None and len(syn) > synthetic_n:
            rng = random.Random(seed)
            syn = rng.sample(syn, synthetic_n)
        items.extend(syn)
    return items


def run(settings: Settings, run_id: str, sets: list[str], modes: list[str], synthetic_n: int | None, seed: int,
        rpm: float | None, judge_model: str | None, judge_n: int, description: str = "", judge_sets: tuple[str, ...] = ("human",)) -> dict[str, Any]:
    api_key = settings.gemini_api_key
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (backend/.env)")
    svc = make_service(settings)
    corpus_ids = corpus_video_ids()
    index_info = ensure_index(svc, corpus_ids)
    gold = select_queries(sets, synthetic_n, seed)
    log.info("[%s] %d queries, modes=%s", run_id, len(gold), modes)

    answerer = CachedAnswerer(CachedGemini(GeminiGenerator(api_key, settings.generation), rpm=rpm))
    pricing = load_pricing()
    price = price_for(settings.generation.model, pricing)
    prompts: dict[tuple[str, str], Prompt] = {}
    records: list[dict[str, Any]] = []

    for mode in modes:
        for g in gold:
            video_ids = corpus_ids if mode == "corpus" else [g.video_id]
            t0 = time.perf_counter()
            capturing = _CapturingAnswerer(answerer)
            ans = svc.answer(g.question, video_ids, api_key, generator=capturing)
            wall = time.perf_counter() - t0
            prompt = capturing.prompt
            if prompt is not None:
                prompts[(g.qid, mode)] = prompt
            cite_secs = [s for _, s in ans.citations]
            spans = [(s.start, s.end) for s in g.spans]
            rec: dict[str, Any] = {
                "qid": g.qid, "set": g.set, "mode": mode, "answerable": g.answerable, "video_id": g.video_id,
                "question": g.question, "answer": ans.text, "abstained": ans.abstained,
                "false_abstention": g.answerable and ans.abstained,
                "missed_abstention": (not g.answerable) and (not ans.abstained),
                "n_sources": len(ans.sources),
                "n_citations": len(ans.citations),
                "citation_validity": citation_validity([lbl for lbl, _ in ans.citations], set(prompt.valid_timestamps) if prompt else set()),
                "citation_localized": citation_localized(cite_secs, spans) if g.answerable else None,
                "invalid_citations": list(ans.invalid_citations),
                "usage": ans.usage.to_dict(),
                "cost_usd": estimate_cost_usd(ans.usage.to_dict(), price),
                "gemini_latency_s": round(ans.latency_s, 3),
                "wall_latency_s": round(wall, 3),
                "context_seconds": ans.context_seconds,
                "prompt_chars": ans.prompt_chars,
                "model_version": ans.model_version,
                "retrieval": ans.extra.get("retrieval"),
                "tags": dict(g.tags),
            }
            records.append(rec)
        log.info("[%s] mode=%s done; cache %s", run_id, mode, answerer.cached.stats)

    judge_info = None
    if judge_model:
        # Judging is a separate, optional pass over answers that have already
        # been paid for. If the judge model is unavailable or rate-limited, keep
        # the run: the deterministic metrics stand on their own and the answers
        # are cached, so judging can be retried without regenerating anything.
        try:
            judge_info = judge(records, prompts, gold, api_key, judge_model, rpm, run_id, judge_n, seed, tuple(judge_sets))
        except Exception as e:
            log.error("Judge pass failed (%s); writing results without judge metrics", e)
            judge_info = {"model": judge_model, "sets": list(judge_sets), "error": f"{type(e).__name__}: {e}", "n_judged": 0}

    metrics = summarize(records, index_info, judge_info, price, pricing, run_id, description)
    out = RESULTS_DIR / run_id
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "per_query.jsonl", records)
    write_json(out / "manifest.json", build_manifest(settings, sets=sets, seed=seed, extra={"kind": "generation", "modes": modes, "judge_model": judge_model, "synthetic_n": synthetic_n}))
    _print(metrics)
    return metrics


def judge(records, prompts, gold, api_key, judge_model, rpm, run_id, judge_n, seed, judge_sets=("human",)) -> dict[str, Any]:
    """Grade faithfulness and correctness on ``judge_sets`` only. Judging every
    answer in every config is the single largest cost in the harness and buys
    little: the human set is the headline, and judge verdicts are only as good
    as the human agreement measured on judge_sample.jsonl anyway."""
    by_qid = {g.qid: g for g in gold}
    # thinking_budget is left at the provider default: Gemini 3 models reject
    # budget 0 outright ("This model only works in thinking mode").
    jg = CachedGemini(GeminiGenerator(api_key, GenerationConfig(model=judge_model, temperature=0.0)), rpm=rpm)
    for rec in records:
        if not rec["answerable"] or rec["abstained"] or rec["set"] not in judge_sets:
            continue
        prompt = prompts.get((rec["qid"], rec["mode"]))
        if prompt is None:
            continue
        context = "\n\n".join(b.render() for b in prompt.blocks)
        f = jg.complete(JUDGE_FAITHFUL_SYSTEM, f"Context:\n{context}\n\nAnswer:\n{rec['answer']}", tag="judge_faithful")
        rec["judge_faithful"] = _parse_verdict(f.text)
        g = by_qid[rec["qid"]]
        if g.short_answer:
            c = jg.complete(JUDGE_CORRECT_SYSTEM, f"Question: {g.question}\nReference answer: {g.short_answer}\nAnswer:\n{rec['answer']}", tag="judge_correct")
            rec["judge_correct"] = _parse_verdict(c.text)
    judged = [r for r in records if "judge_faithful" in r]
    rng = random.Random(seed)
    sample = rng.sample(judged, min(judge_n, len(judged)))
    rows = [
        {
            "qid": r["qid"], "mode": r["mode"], "question": r["question"], "answer": r["answer"],
            "context": "\n\n".join(b.render() for b in prompts[(r["qid"], r["mode"])].blocks),
            "judge_faithful": r["judge_faithful"], "judge_correct": r.get("judge_correct"),
            "human_faithful": "", "human_correct": "",
        }
        for r in sample
    ]
    write_jsonl(RESULTS_DIR / run_id / "judge_sample.jsonl", rows)
    log.info("Judge: %d items judged with %s; %d written for human agreement", len(judged), judge_model, len(rows))
    return {"model": judge_model, "n_judged": len(judged), "sample_n": len(rows), "cache": jg.stats}


def summarize(records, index_info, judge_info, price, pricing, run_id, description) -> dict[str, Any]:
    def block(recs: list[dict[str, Any]]) -> dict[str, Any]:
        ans = [r for r in recs if r["answerable"]]
        una = [r for r in recs if not r["answerable"]]
        validity = [r["citation_validity"] for r in ans if r["citation_validity"] is not None]
        localized = [r["citation_localized"] for r in ans if r["citation_localized"] is not None]
        out: dict[str, Any] = {
            "n": len(recs), "n_answerable": len(ans), "n_unanswerable": len(una),
            "abstention_on_unanswerable": (sum(r["abstained"] for r in una) / len(una)) if una else None,
            "false_abstention_on_answerable": (sum(r["abstained"] for r in ans) / len(ans)) if ans else None,
            "answers_with_citations": (sum(1 for r in ans if r["n_citations"] > 0 and not r["abstained"]) / len(ans)) if ans else None,
            "citation_validity_mean": mean(validity) if validity else None,
            "citation_localized_rate": (sum(localized) / len(localized)) if localized else None,
            "gemini_latency_s": {"p50": percentile([r["gemini_latency_s"] for r in recs], 50), "p95": percentile([r["gemini_latency_s"] for r in recs], 95)},
            "tokens_mean": {k: mean(r["usage"][k] for r in recs) for k in ("prompt_tokens", "output_tokens", "thinking_tokens")} if recs else {},
            "cost_usd_mean": mean(r["cost_usd"] for r in recs) if recs else None,
            "cost_usd_total": sum(r["cost_usd"] for r in recs),
            "context_seconds_mean": mean(r["context_seconds"] for r in recs) if recs else None,
        }
        for key in ("judge_faithful", "judge_correct"):
            vals = [r[key] for r in ans if key in r]
            if vals:
                out[key] = {v: vals.count(v) / len(vals) for v in sorted(set(vals))}
                out[f"{key}_n"] = len(vals)
        return out

    metrics: dict[str, Any] = {"run_id": run_id, "description": description, "index": index_info, "pricing": {**price, "as_of": pricing.get("as_of")}, "judge": judge_info, "by_mode": {}}
    for mode in sorted({r["mode"] for r in records}):
        mrecs = [r for r in records if r["mode"] == mode]
        metrics["by_mode"][mode] = {"all": block(mrecs), "by_set": {s: block([r for r in mrecs if r["set"] == s]) for s in sorted({r["set"] for r in mrecs})}}
    return metrics


def _print(m: dict[str, Any]) -> None:
    print(f"\n== {m['run_id']} ==")
    for mode, blk in m["by_mode"].items():
        a = blk["all"]
        print(
            f"  {mode:6s} n={a['n']}  abstain@unans={a['abstention_on_unanswerable']}  false_abstain={a['false_abstention_on_answerable']}  "
            f"cite_valid={a['citation_validity_mean']}  cite_local={a['citation_localized_rate']}  "
            f"lat_p50={a['gemini_latency_s']['p50']:.2f}s  cost/q=${a['cost_usd_mean'] or 0:.5f}"
        )
        for key in ("judge_faithful", "judge_correct"):
            if key in a:
                print(f"         {key}: {a[key]} (n={a[key + '_n']})")


def main() -> None:
    load_env()
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--run-id")
    ap.add_argument("--sets", nargs="+", default=["human", "unanswerable", "synthetic"])
    ap.add_argument("--modes", nargs="+", default=["single"], choices=["single", "corpus"])
    ap.add_argument("--synthetic-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rpm", type=float, default=None)
    ap.add_argument("--judge", default=None, help="judge model name, e.g. gemini-2.5-pro; omit to skip")
    ap.add_argument("--judge-n", type=int, default=30, help="items written for human agreement")
    ap.add_argument("--judge-sets", nargs="+", default=["human"], choices=["human", "synthetic"], help="which sets to judge")
    args = ap.parse_args()
    if not args.config:
        ap.error("--config required")
    name, settings, raw = load_config(args.config)
    run(settings, args.run_id or f"G_{name}", args.sets, args.modes, args.synthetic_n, args.seed, args.rpm, args.judge, args.judge_n, raw.get("description", ""), tuple(args.judge_sets))


if __name__ == "__main__":
    main()
