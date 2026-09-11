# Lectura

Ask questions about a lecture video and get an answer that cites the exact moment it came from —
click a timestamp and the player seeks there.

Every default in the shipped configuration was chosen by the evaluation harness in
[`backend/eval/`](backend/eval/README.md), against decision rules fixed before the experiments ran.
Full tables: [`backend/eval/RESULTS.md`](backend/eval/RESULTS.md).

## Pipeline

```
YouTube URL ──► captions (manual preferred) ──► none? yt-dlp audio ──► Whisper (local)
transcript ──► 30s chunks ──► bge-small-en-v1.5 ──► ChromaDB (cosine)
question ──► top-20 candidates ──► cross-encoder rerank ──► top-4
         ──► 120s parent windows rebuilt from the transcript
         ──► Gemini 2.5 Flash, told to cite only the [MM:SS] labels it was given
         ──► answer + citations validated against those labels
```

## Results

Benchmark: 11 professionally captioned lectures (3Blue1Brown, Stanford CS229, CS231n), 9.5 hours,
1,138 chunks. Questions: 60 human-verified with evidence spans, 223 filtered synthetic, 30
unanswerable.

Retrieval, tuned configuration vs the one the project started with, over 283 answerable questions:

| | Hit@1 | Hit@3 | MRR@10 |
|---|---|---|---|
| Single video, before | 32.9% | 60.8% | 0.491 |
| Single video, **after** | **52.3%** | **72.1%** | **0.630** |
| Whole corpus, before | 27.6% | 47.3% | 0.403 |
| Whole corpus, **after** | **48.4%** | **66.8%** | **0.583** |

Paired bootstrap, 2000 resamples: **+0.146 MRR@10** on synthetic (95% CI [+0.099, +0.194]) and
**+0.254** on the human set (CI [+0.152, +0.354]).

End-to-end on the 60 human questions — an answer counts only if the system did not refuse *and* a
judge called it correct: **91.7% → 96.7%**. The judge was itself audited blind against a person on
37 answers (97.3% agreement, erring strict). Citation validity is **99.2%**, checked
deterministically against the exact timestamps supplied in each prompt.

Transcription: Whisper `tiny.en` reaches 7.9% WER at 0.037 real-time factor, and rebuilding the
index from Whisper transcripts instead of captions costs no measurable retrieval quality — every
paired CI includes zero. Hence `tiny.en` as the default.

Six candidate improvements were measured and **rejected**, including 60-second chunks, whose MRR
gain proved to be an artefact of coarser granularity: on an equal context budget it scored −0.049
and tripled citation error. See [`RESULTS.md`](backend/eval/RESULTS.md) for all of them.

## Running it

Python 3.11+ (3.13 tested), Node 18+ (22 tested), FFmpeg on PATH for the Whisper fallback.

```bash
cd backend
python -m venv venv && .\venv\Scripts\activate     # macOS/Linux: source venv/bin/activate
# optional GPU: pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0
pip install -r requirements.txt
python main.py                                      # http://localhost:8000, docs at /docs
```

```bash
cd frontend && npm install && npm run dev           # http://localhost:5173
```

Paste a Google AI Studio key via the key icon, then a YouTube link. Videos move
`processing` → `transcribing` → `indexed`, or `failed` with a retry. The key travels per request in
the `X-Gemini-API-Key` header and is never stored server-side; there is no authentication, so keep
the server on localhost.

```bash
cd backend  && pytest && ruff check .               # 186 tests, no network or model downloads
cd frontend && npm run build && npm run lint
```

Settings are `LECTURA_`-prefixed environment variables — see `backend/.env.example`. Anything that
changes the vectors or chunk boundaries derives a new collection name automatically, so experiments
never collide with your library.

## Layout

```
backend/
  lectura/     config, catalog, transcripts, chunking, embeddings, index,
               retrieval, generation, service, api
  tests/       186 tests: temp dirs, a hashed test embedder, faked LLM
  eval/        benchmark, gold sets, runners, report -> RESULTS.md
frontend/      React + TypeScript + Vite
```
