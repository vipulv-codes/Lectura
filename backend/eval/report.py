"""Render eval/RESULTS.md from everything under eval/results/.

    python -m eval.report [--baseline baseline]

Sections: retrieval configs, Stage A marginals, paired comparisons against the
baseline (bootstrap CIs), gate calibration, ANN-vs-exact, generation, ASR,
negative results, reproducibility. Every run that exists is listed; nothing is
filtered out.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import EVAL_DIR, RESULTS_DIR, read_json, read_jsonl
from .metrics import paired_bootstrap

OUT = EVAL_DIR / "RESULTS.md"


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int | float):
        if isinstance(x, float) and math.isnan(x):
            return "-"
        return f"{x:.{digits}f}" if isinstance(x, float) else str(x)
    return str(x)


def pct(x: Any) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_no runs_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def load_runs() -> dict[str, dict[str, Any]]:
    runs = {}
    if not RESULTS_DIR.exists():
        return runs
    for d in sorted(RESULTS_DIR.iterdir()):
        mp = d / "metrics.json"
        if d.is_dir() and mp.exists():
            m = read_json(mp)
            man = read_json(d / "manifest.json") if (d / "manifest.json").exists() else {}
            runs[d.name] = {"metrics": m, "manifest": man, "dir": d}
    return runs


def retrieval_rows(runs, mode: str, set_name: str | None) -> list[list[str]]:
    rows = []
    for rid, r in runs.items():
        m = r["metrics"]
        if r["manifest"].get("kind") != "retrieval" or mode not in m.get("by_mode", {}):
            continue
        blk = m["by_mode"][mode]
        a = blk["by_set"].get(set_name) if set_name else blk["all_answerable"]
        if not a:
            continue
        idx = m["index"]
        child = r["manifest"].get("config", {}).get("index", {}).get("child_secs", "")
        rows.append([
            rid, str(child),
            str(idx.get("n_chunks")), str(a["n"]), pct(a["hit@1"]), pct(a["hit@3"]), fmt(a["mrr@10"]),
            pct(a["hit@120s"]), pct(a["hit@240s"]), fmt(a["ts_error"].get("median"), 1), pct(a["ts_error"].get("frac_within_10s")),
            f"{(a.get('latency_s') or {}).get('p50', float('nan')) * 1000:.0f}",
        ])
    return rows


RET_HEADERS = ["run", "child s", "chunks", "n", "Hit@1", "Hit@3", "MRR@10", "Hit@120s", "Hit@240s", "ts err med (s)", "ts err ≤10s", "p50 ms"]


def marginals(runs, mode: str, set_name: str) -> str:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rid, r in runs.items():
        if not rid.startswith("A_") or r["manifest"].get("kind") != "retrieval":
            continue
        a = r["metrics"].get("by_mode", {}).get(mode, {}).get("by_set", {}).get(set_name)
        if not a:
            continue
        cfg = r["manifest"]["config"]
        axes = {
            "embedder": cfg["index"]["embedding_model"].split("/")[-1],
            "child_secs": str(cfg["index"]["child_secs"]),
            "strategy": cfg["retrieval"]["strategy"],
            "reranker": "yes" if cfg["retrieval"].get("reranker") else "no",
        }
        for axis, val in axes.items():
            groups[axis][val].append(a["mrr@10"])
            groups[axis + "|hit120"][val].append(a["hit@120s"])
    if not groups:
        return "_Stage A grid not run yet_\n"
    out = []
    for axis in ("embedder", "child_secs", "strategy", "reranker"):
        rows = []
        for val in sorted(groups[axis]):
            m = groups[axis][val]
            h = groups[axis + "|hit120"][val]
            rows.append([val, str(len(m)), fmt(sum(m) / len(m)), fmt(min(m)), fmt(max(m)), pct(sum(h) / len(h))])
        out.append(f"**{axis}** (mean over the other axes; {mode} mode, {set_name} set)\n\n" + table([axis, "runs", "mean MRR@10", "min", "max", "mean Hit@120s"], rows))
    return "\n".join(out)


def paired(runs, baseline: str, mode: str, set_name: str) -> str:
    base = runs.get(baseline)
    if not base:
        return f"_baseline run '{baseline}' not found_\n"
    base_pq = {(r["qid"], r["mode"]): r for r in read_jsonl(base["dir"] / "per_query.jsonl")}
    rows = []
    for rid, r in runs.items():
        if rid == baseline or r["manifest"].get("kind") != "retrieval":
            continue
        pq = read_jsonl(r["dir"] / "per_query.jsonl")
        a, b = [], []
        for rec in pq:
            if rec["mode"] != mode or rec["set"] != set_name or not rec["answerable"]:
                continue
            brec = base_pq.get((rec["qid"], rec["mode"]))
            if brec is None:
                continue
            a.append(_rr(brec["first_hit_rank"]))
            b.append(_rr(rec["first_hit_rank"]))
        if not a:
            continue
        res = paired_bootstrap(a, b, 2000, seed=0)
        rows.append([rid, str(res.n), f"{res.mean_diff:+.3f}", f"[{res.ci_low:+.3f}, {res.ci_high:+.3f}]", f"{res.wins}/{res.losses}/{res.ties}", "yes" if res.significant else "no"])
    return table(["run", "n", "Δ MRR@10 vs baseline", "95% CI", "win/loss/tie", "CI excludes 0"], rows)


def _rr(rank: int | None) -> float:
    return 0.0 if rank is None or rank >= 10 else 1.0 / (rank + 1)


def gate_rows(runs, mode: str) -> list[list[str]]:
    rows = []
    for rid, r in runs.items():
        g = r["metrics"].get("gate", {}).get(mode) if r["manifest"].get("kind") == "retrieval" else None
        if not g or "auroc" not in g:
            continue
        rows.append([rid, str(g["n_pos"]), str(g["n_neg"]), fmt(g["auroc"]), fmt(g["threshold_at_95tpr"]), pct(g["tpr_at_threshold"]), pct(g["fpr_at_threshold"]), fmt(g["pos_score_median"]), fmt(g["neg_score_median"])])
    return rows


def generation_rows(runs, mode: str) -> tuple[list[list[str]], list[list[str]]]:
    rows, jrows = [], []
    for rid, r in runs.items():
        if r["manifest"].get("kind") != "generation":
            continue
        blk = r["metrics"].get("by_mode", {}).get(mode)
        if not blk:
            continue
        a = blk["all"]
        rows.append([rid, str(a["n"]), pct(a["abstention_on_unanswerable"]), pct(a["false_abstention_on_answerable"]), pct(a["answers_with_citations"]), pct(a["citation_validity_mean"]), pct(a["citation_localized_rate"]),
                     f"{a['gemini_latency_s']['p50']:.2f}", f"{a['tokens_mean'].get('prompt_tokens', 0):.0f}/{a['tokens_mean'].get('output_tokens', 0):.0f}/{a['tokens_mean'].get('thinking_tokens', 0):.0f}", f"${a['cost_usd_mean'] or 0:.5f}"])
        if "judge_faithful" in a:
            jf, jc = a["judge_faithful"], a.get("judge_correct", {})
            jrows.append([rid, r["metrics"].get("judge", {}).get("model", ""), str(a["judge_faithful_n"]), pct(jf.get("supported", 0)), pct(jf.get("partially_supported", 0)), pct(jf.get("unsupported", 0)), pct(jc.get("correct", 0)), pct(jc.get("partially_correct", 0)), pct(jc.get("incorrect", 0))])
    return rows, jrows


def asr_section(runs) -> str:
    asr = runs.get("asr")
    if not asr:
        return "_not run yet_\n"
    m = asr["metrics"]
    rows = [[model, str(s.get("n_videos")), fmt(s.get("audio_hours"), 2), pct(s.get("pooled_wer")), pct(s.get("mean_wer")), fmt(s.get("mean_rtf")), str(s.get("hallucination_loops"))] for model, s in m["summary"].items()]
    out = table(["model", "videos", "audio h", "pooled WER", "mean WER", "GPU RTF", "loops"], rows)
    if m.get("cpu_rtf"):
        out += "\nReal-time factor on a fixed clip:\n\n" + table(["model", "device", "clip s", "RTF"], [[r["model"], r["device"], str(r["clip_seconds"]), fmt(r["rtf"])] for r in m["cpu_rtf"]])
    out += f"\n> {m.get('caveat', '')}\n"
    return out


def dataset_status() -> str:
    """One line per gold set: size and whether a person has verified it."""
    from .common import GOLD_FILES
    from .gold import load_jsonl

    lines = ["**Datasets**", ""]
    for name, path in GOLD_FILES.items():
        if not path.exists():
            lines.append(f"- `{name}`: missing")
            continue
        items = load_jsonl(path)
        if name == "synthetic":
            lines.append(f"- `{name}`: {len(items)} items (model-generated, filtered; see README)")
            continue
        unverified = sum(1 for it in items if not it.tags.get("verified", False))
        state = "**verified by a person**" if unverified == 0 else f"**{unverified} of {len(items)} still unverified (LLM-drafted)**"
        lines.append(f"- `{name}`: {len(items)} items, {state}")
    return "\n".join(lines)


def repro_section(runs) -> str:
    rows = []
    for rid, r in runs.items():
        man = r["manifest"]
        if not man:
            continue
        pk = man.get("packages", {})
        rows.append([rid, man.get("kind", ""), (man.get("git_sha") or "")[:8], man.get("created", ""), man.get("device", {}).get("gpu", "cpu"), pk.get("chromadb", ""), pk.get("sentence-transformers", ""), pk.get("torch", "")])
    return table(["run", "kind", "git", "created", "device", "chromadb", "s-t", "torch"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    runs = load_runs()
    parts = ["# Results", "", "Generated by `python -m eval.report`. Every run under `eval/results/` is listed; see `eval/README.md` for the protocol, decision rules and caveats.", ""]
    parts += [dataset_status(), ""]

    parts += ["## 1. Retrieval", "", "### Human set, single-video mode (the product's use case)", "", table(RET_HEADERS, retrieval_rows(runs, "single", "human")),
              "### Human set, corpus mode (all videos searchable; primary decision metric = MRR@10)", "", table(RET_HEADERS, retrieval_rows(runs, "corpus", "human")),
              "### Synthetic set, corpus mode", "", table(RET_HEADERS, retrieval_rows(runs, "corpus", "synthetic")),
              "### Synthetic set, single-video mode", "", table(RET_HEADERS, retrieval_rows(runs, "single", "synthetic"))]
    parts += ["## 2. Stage A marginals", "", marginals(runs, "corpus", "synthetic"), "", marginals(runs, "corpus", "human")]
    parts += ["## 3. Paired comparison against the baseline (reciprocal rank @10, 2000-resample bootstrap)", "",
              "Synthetic, corpus mode:", "", paired(runs, args.baseline, "corpus", "synthetic"), "Human, corpus mode:", "", paired(runs, args.baseline, "corpus", "human"), "Human, single mode:", "", paired(runs, args.baseline, "single", "human")]
    parts += ["## 4. Relevance gate calibration (best candidate score, answerable vs unanswerable)", "",
              "Single-video mode:", "", table(["run", "n+", "n-", "AUROC", "thr @95% TPR", "TPR", "FPR", "median + score", "median - score"], gate_rows(runs, "single")),
              "Corpus mode:", "", table(["run", "n+", "n-", "AUROC", "thr @95% TPR", "TPR", "FPR", "median + score", "median - score"], gate_rows(runs, "corpus"))]
    ann = [[rid, str(r["metrics"]["ann_vs_exact"]["n_vectors"]), str(r["metrics"]["ann_vs_exact"]["k"]), pct(r["metrics"]["ann_vs_exact"]["recall"])] for rid, r in runs.items() if r["metrics"].get("ann_vs_exact")]
    parts += ["## 5. HNSW vs exact search", "", table(["run", "vectors", "k", "recall"], ann)]
    grows, jrows = generation_rows(runs, "single")
    parts += ["## 6. Generation (single-video mode)", "", table(["run", "n", "abstain on unanswerable", "false abstain", "answers w/ citations", "citation validity", "citation localised", "p50 s", "tokens in/out/think", "cost/query"], grows),
              "LLM-judge (indicative; see judge agreement in README):", "", table(["run", "judge", "n", "supported", "partial", "unsupported", "correct", "partial", "incorrect"], jrows)]
    agreement_file = RESULTS_DIR / "judge_agreement.json"
    if agreement_file.exists():
        ag = read_json(agreement_file)
        n = ag.get("n_graded", 0)
        scored = ", ".join(
            f"{k} {v['agreement']:.0%}" + (f" (κ={v['kappa']:.2f})" if ag.get("kappa_reliable") else "")
            for k, v in ag.items()
            if isinstance(v, dict) and "agreement" in v
        )
        line = f"Judge audited blind against a person on **{n} answers** ({ag.get('run', '?')}): {scored}."
        if ag.get("judge_correct_precision") is not None:
            line += (f" Of {ag['n_judge_called_correct']} answers it called correct, a person agreed with "
                     f"{round(ag['judge_correct_precision'] * ag['n_judge_called_correct'])}.")
        if not ag.get("kappa_reliable", True):
            line += " Kappa is omitted: a rater gave the same verdict throughout, so chance agreement is ~100%."
        parts += [line, ""]
    parts += ["## 7. Transcription (Whisper vs manual captions)", "", asr_section(runs)]
    parts += ["## 8. Negative results", "", "Runs whose paired CI against the baseline includes zero or is negative stay in the tables above; nothing is dropped. See section 3.", ""]
    parts += ["## 9. Reproducibility", "", repro_section(runs)]
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
