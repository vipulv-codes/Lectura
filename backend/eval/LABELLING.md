# Verifying the gold sets

`data/human.jsonl` holds 69 questions **drafted by an LLM** from transcript windows, and
`data/unanswerable.jsonl` holds 16 hand-written in-domain questions the lectures should not
answer. Every item is tagged `"verified": false`. The headline numbers stay provisional until a
person has checked them. Budget: 1 to 2 hours.

## Use the labeller

```bash
cd backend && .\venv\Scripts\activate
python -m eval.label                    # walks every unverified human item
python -m eval.label --set unanswerable
python -m eval.label --qid h_001        # jump to one item
```

It shows the question, the current span, and the surrounding transcript with the span marked `>`.
You type the evidence timestamps **as they appear on screen** and it snaps them to real caption
boundaries:

```
> 02:52 02:58
  snapped -> [02:52]-[03:00]  172.50-180.44  (7.9s)
  verified.
```

Commands: `Enter` keeps the current span, `ans <text>` / `qn <text>` fix the wording, `w` widens
the context shown, `s` skips, `d` deletes, `x` saves and exits, `?` shows help. The file is
written after every item, so you can stop and resume whenever.

## What you are deciding

For each item, read the window and ask:

1. Would a student who watched the lecture actually ask this, without copying the lecturer's words?
2. Do the lines inside the span genuinely support the short answer?
3. Is the span tight: does it start at the first supporting line and end at the last?

Delete anything that does not hold up. A smaller verified set beats a larger drafted one. Add a
second span (`02:52 02:58, 10:20 10:31`) when the lecturer answers the same question twice.

**Do not check by asking the app.** That biases the set toward what the retriever already finds,
which is the thing these labels are meant to measure.

### The span convention

A span runs from the **start of the first evidence line** to the **end of the last one**. That is
what `make_synthetic.py` does, so the human and synthetic sets stay comparable. The tool enforces
it, so you only ever type the two labels.

Sub-second precision does not matter much: chunks are 30 s and a hit needs 5 s of overlap. The
**start** is the part worth getting right, because `anchor` is what the timestamp-error metric
measures, and it is the moment a student would be sent to.

## Unanswerable items

`python -m eval.label --set unanswerable` shows the question and its assigned video. Confirm the
video really does not cover that topic; `/word` greps its transcript so you can check quickly
(e.g. `/dropout`). Press `y` if it is genuinely not covered, `d` if it is. About 15 minutes.

The `video_swap` negatives need no checking; regenerate them from the verified human set with:

```bash
python -m eval.make_synthetic swaps --n 15
```

## Afterwards

```bash
python -m eval.run_retrieval --config baseline
python -m eval.report
```

`RESULTS.md` flips the dataset line from "still unverified" to "verified by a person" once every
item carries `verified: true`.
