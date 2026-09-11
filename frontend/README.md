# Lectura — Frontend

React + TypeScript + Vite single-page app for Lectura. It talks to the FastAPI backend
(`http://localhost:8000` by default) for ingestion, RAG queries, and chapter generation.

## Scripts

```bash
npm install      # install dependencies
npm run dev      # start the dev server (http://localhost:5173)
npm run build    # type-check and build for production
npm run preview  # preview the production build
npm run typecheck
npm run lint     # ESLint, zero warnings allowed
```

## Structure

- `src/App.tsx` — workspace state, library polling, persistence
- `src/components/` — Sidebar, VideoPlayer, Chat, Chapters, KeyInput
- `src/lib/api.ts` — base URL and the fetch wrapper that normalises backend errors
- `src/lib/storage.ts` — guarded localStorage access
- `src/lib/format.ts` — duration formatting and the video status type
- `src/App.css` — styling

Set `VITE_API_BASE_URL` in a `.env` file to point at a backend that is not on
`localhost:8000`. See `NOTES.md` for the current state of the frontend.
