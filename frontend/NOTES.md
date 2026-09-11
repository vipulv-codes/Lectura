# Frontend notes

Status as of 2026-09-08. The frontend received correctness fixes only; whether
to keep it, slim it down, or drop it in favour of a CLI/notebook demo is an
open decision. This file records what was fixed and what was deliberately left.

## Fixed

- A `watch?v=…&list=…` URL used to ingest the whole playlist. Ingest now goes to
  one endpoint and the backend decides (video wins when `v=` is present).
- FastAPI validation errors (`detail` as an array) rendered as `[object Object]`.
  All requests go through `lib/api.ts`, which normalises error bodies.
- Backend LLM failures came back as HTTP 200 text and were cached as the chapter
  outline. The backend now returns 401/429/502 and the UI never persists failures.
- Chat histories and chapter outlines were updated from stale render snapshots
  (deleting a video while a query was in flight resurrected its history).
  All updates use functional setters; persistence is an effect on the state.
- `Chat` and `Chapters` are keyed by the active id so loading state and drafts
  no longer leak across lecture switches.
- `VideoPlayer` leaked the YouTube player when the IFrame API loaded late and
  grew the `onYouTubeIframeAPIReady` chain on every mount. Seeks are now an
  explicit `{seconds, nonce}` request instead of a `Math.random()` jitter.
- Library polling recreated its interval on every tick and had no ordering
  guard; it now keys on a boolean and drops out-of-order responses.
- Sidebar timers are cleared on unmount and on each submit.
- `.spinner` forced a 12px bordered box onto lucide icons; the ring is now
  `.spinner-ring`, and `.spinner` only animates.
- `strict` TypeScript, `--max-warnings 0` lint, all storage access wrapped.
- Retrieved `snippet` text is no longer persisted to localStorage.

## Left as is (decide with the keep-or-scrap question)

- Markdown/citation parsing is duplicated between `Chat.tsx` and
  `Chapters.tsx` (~120 lines each). Both are pure functions and would be the
  first thing to unit-test if the frontend stays.
- Roughly a quarter of `App.css` is hover/transition/scrollbar decoration;
  `.chapters-container ul` can never match (list items are rendered without a
  wrapping `<ul>`), `--radius-lg` is unused, `.download` and `.folder-icon` have
  no rules, `.chat-header`/`.chapters-header` and their buttons are duplicated.
- No frontend tests and no type-aware ESLint preset.
- The YouTube IFrame API is loaded from a CDN at runtime (offline = no player).
- The Gemini key lives in localStorage in plain text; any XSS could read it.
