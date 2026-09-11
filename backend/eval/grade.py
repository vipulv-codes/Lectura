"""Blind human audit of the LLM judge, to test whether it rubber-stamps.

    python -m eval.grade --run G1_tuned            # grade the 30-item sample
    python -m eval.grade --run G1_tuned --report   # agreement only, no grading
    python -m eval.grade --run G1_tuned --redo h_055   # clear and re-grade items

The judge's own verdict is hidden until after you commit to yours: seeing it
first turns the exercise into agreeing with a machine.

Why this matters here: the judge returned "supported"/"correct" for 98% of
judgements, so Cohen's kappa against it is close to meaningless (a rater who
always says "correct" scores near-perfect agreement). The number that carries
information is the precision of its positive verdicts -- of the answers it
blessed, how many does a person agree with. The report leads on that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from lectura.chunking import find_timestamps
from lectura.generation import build_prompt

from .common import CONFIGS_DIR, RESULTS_DIR, load_config, make_service, write_json
from .metrics import agreement, cohen_kappa

FAITHFUL = {"y": "supported", "p": "partially_supported", "n": "unsupported"}
CORRECT = {"y": "correct", "p": "partially_correct", "n": "incorrect"}

# Separator between context blocks, matching how run_generation renders them.
SEP = "\n\n"

HELP = """
  y = yes, fully      p = partly      n = no
  s = skip this item  x = save and exit      ? = show this help again

  Q1 "does every fact come from the context?"
     Did the model stick to the transcript, or add things of its own? Every
     factual claim in the ANSWER should trace to a line in the CONTEXT below;
     lines the answer cites are marked with * so they are easy to find.
     An answer can be perfectly true in the real world and still fail this, if
     it came from the model's own knowledge rather than the lecture. Catching
     that is the entire point of the exercise.
       y = every claim is in the context
       p = the main point is, but some detail is not
       n = the main point is not in the context at all

  Q2 "does it answer the question asked?"
     Ignoring where the material came from, is it a correct answer?
       y = yes        p = partly, or it drifts off the question        n = no
"""


def _settings_for(run: str):
    """The config a run was produced with. Run ids and config names mostly
    coincide (G1_tuned), but not always (G0_baseline used baseline.yaml), so
    try the run id first and fall back to the suffix after the stage prefix."""
    for name in (run, run.split("_", 1)[-1]):
        if (CONFIGS_DIR / f"{name}.yaml").exists():
            return load_config(name)[1]
    raise SystemExit(f"no config found for run {run}")


def build_hard_sample(runs: list[str], out_run: str = "judge_hard") -> Path:
    """Collect every answer the judge did NOT give a clean pass, across runs.

    A random sample drawn from a pool that is 96% clean passes contains none of
    the judge's negative verdicts, so it can show that its positives are sound
    while saying nothing about whether its negatives are. These are the items
    where it made a distinction, so they are the ones worth checking. The
    context is rebuilt by re-running retrieval with that run's own config; the
    answers are the stored ones, so nothing is regenerated.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for run in runs:
        pq = RESULTS_DIR / run / "per_query.jsonl"
        if not pq.exists():
            continue
        recs = [json.loads(x) for x in pq.read_text(encoding="utf-8").splitlines() if x.strip()]
        hard = [
            r for r in recs
            if "judge_faithful" in r
            and (r["judge_faithful"] != "supported" or r.get("judge_correct") not in (None, "correct"))
        ]
        if not hard:
            continue
        settings = _settings_for(run)
        svc = make_service(settings)
        for r in hard:
            key = (run, r["qid"])
            if key in seen:
                continue
            seen.add(key)
            res = svc.retrieve(r["question"], [r["video_id"]])
            prompt = build_prompt(
                r["question"], res.hits, svc.transcripts,
                svc.titles([h.chunk.video_id for h in res.hits]),
                settings.retrieval.parent_secs, settings.generation.context_order,
            )
            rows.append({
                "qid": f"{run}:{r['qid']}", "mode": r["mode"], "question": r["question"],
                "answer": r["answer"],
                "context": SEP.join(b.render() for b in prompt.blocks),
                "judge_faithful": r["judge_faithful"], "judge_correct": r.get("judge_correct"),
                "human_faithful": "", "human_correct": "",
            })
    out = RESULTS_DIR / out_run / "judge_sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, rows)
    print(f"wrote {len(rows)} hard case(s) to {out}")
    return out


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def ask(prompt: str, mapping: dict[str, str]) -> str | None:
    while True:
        raw = input(prompt).strip().lower()
        if raw in mapping:
            return mapping[raw]
        if raw == "?":
            print(HELP)
        elif raw in ("s", "x"):
            return raw
        else:
            print("  y = yes / p = partly / n = no   (? for help, s to skip, x to exit)")


_TS_LINE = re.compile(r"\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def show_context(context: str, answer: str) -> None:
    """Print the whole retrieved context, marking the lines the answer cites.

    Shown in full and never truncated: the context is exactly what the model
    was given, so checking a claim means finding it here, and hiding part of it
    would conceal the evidence the question is about.
    """
    cited = {label for label, _ in find_timestamps(answer)}
    for line in context.splitlines():
        m = _TS_LINE.match(line)
        mark = "*" if m and m.group(1) in cited else " "
        print(f"  {mark} {line}")


def report(rows: list[dict[str, Any]], run: str) -> dict[str, Any]:
    done = [r for r in rows if r.get("human_faithful") and r.get("human_correct")]
    if not done:
        print(f"{run}: nothing graded yet")
        return {}
    jf = [r["judge_faithful"] for r in done]
    hf = [r["human_faithful"] for r in done]
    jc = [r["judge_correct"] for r in done]
    hc = [r["human_correct"] for r in done]
    pos = [r for r in done if r["judge_correct"] == "correct"]
    pos_ok = sum(1 for r in pos if r["human_correct"] == "correct")
    out = {
        "run": run,
        "n_graded": len(done),
        "faithful": {"agreement": agreement(jf, hf), "kappa": cohen_kappa(jf, hf)},
        "correct": {"agreement": agreement(jc, hc), "kappa": cohen_kappa(jc, hc)},
        "judge_correct_precision": (pos_ok / len(pos)) if pos else None,
        "n_judge_called_correct": len(pos),
        "human_disagreed_on": [r["qid"] for r in done if r["judge_correct"] != r["human_correct"]],
    }
    # Kappa corrects for chance agreement, which needs both raters to actually
    # vary. On a sample where one of them gives the same verdict throughout,
    # expected agreement is ~1 and kappa is meaningless rather than excellent.
    degenerate = len(set(jc)) < 2 or len(set(hc)) < 2
    out["kappa_reliable"] = not degenerate

    print(f"\n=== {run}: {len(done)} items graded ===")
    print(f"  faithfulness  agreement {out['faithful']['agreement']:.1%}   kappa {out['faithful']['kappa']:.2f}")
    print(f"  correctness   agreement {out['correct']['agreement']:.1%}   kappa {out['correct']['kappa']:.2f}")
    if pos:
        print(f"  of {len(pos)} answers the judge called correct, a person agreed with "
              f"{pos_ok} ({out['judge_correct_precision']:.1%})")
    if out["human_disagreed_on"]:
        print(f"  disagreements: {', '.join(out['human_disagreed_on'])}")
    if degenerate:
        print("  note: kappa is NOT meaningful here - a rater gave the same verdict throughout,")
        print("        so chance agreement is ~100%. Quote the agreement rate instead.")
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="G1_tuned", help="results directory holding judge_sample.jsonl")
    ap.add_argument("--report", action="store_true", help="print agreement without grading")
    ap.add_argument("--redo", nargs="*", default=None, help="clear grades for these qids (or all) and grade again")
    ap.add_argument("--build-hard-sample", nargs="*", default=None, metavar="RUN",
                    help="collect the judge's non-clean-pass verdicts from these runs into judge_hard")
    args = ap.parse_args()

    if args.build_hard_sample is not None:
        build_hard_sample(args.build_hard_sample or ["G0_baseline", "G1_tuned", "G3_nothink"])
        return

    path = RESULTS_DIR / args.run / "judge_sample.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    rows = load(path)

    if args.redo is not None:
        targets = set(args.redo)
        n = 0
        for r in rows:
            if not targets or r["qid"] in targets:
                r["human_faithful"] = r["human_correct"] = ""
                n += 1
        save(path, rows)
        print(f"cleared {n} grade(s)")

    if not args.report:
        todo = [i for i, r in enumerate(rows) if not r.get("human_correct")]
        print(f"\n{len(todo)} of {len(rows)} left to grade in {args.run}.")
        print("Two questions per item. The judge's verdict stays hidden until after you answer.")
        print(HELP)
        for n, i in enumerate(todo, 1):
            r = rows[i]
            print("=" * 100)
            print(f"[{n}/{len(todo)}] {r['qid']}")
            # The hard sample holds one row per (config, question), so the same
            # question recurs with a different answer. Say so loudly: it reads
            # as a duplicate otherwise, and the whole point is that the configs
            # answered it differently.
            twins = [o for o in rows if o is not r and o["question"] == r["question"]]
            if twins:
                print(f"\n  NOTE: same question as {', '.join(t['qid'] for t in twins)}, but a"
                      f" DIFFERENT answer from another config. Judge this answer on its own.")
            print(f"\nQUESTION: {r['question']}\n")
            print(f"ANSWER:\n{r['answer']}\n")
            print("CONTEXT the model was given (* marks a line the answer cites):")
            show_context(r["context"], r["answer"])
            f = ask("\n  Q1. Does every fact in the answer come from that context? [y/p/n] > ", FAITHFUL)
            if f in ("s", "x"):
                if f == "x":
                    break
                continue
            c = ask("  Q2. Does it actually answer the question asked?           [y/p/n] > ", CORRECT)
            if c in ("s", "x"):
                if c == "x":
                    break
                continue
            rows[i]["human_faithful"], rows[i]["human_correct"] = f, c
            save(path, rows)
            verdict = "agrees" if c == r["judge_correct"] else f"DISAGREES (it said {r['judge_correct']})"
            print(f"  recorded; the judge {verdict}\n")
        save(path, rows)

    out = report(rows, args.run)
    if out:
        write_json(RESULTS_DIR / "judge_agreement.json", out)
        print(f"\nwrote {RESULTS_DIR / 'judge_agreement.json'}")


if __name__ == "__main__":
    main()
