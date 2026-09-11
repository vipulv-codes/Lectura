"""HTTP layer. ``create_app`` builds a FastAPI app around a ``Lectura`` service
so tests and the eval harness can construct isolated instances."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .config import ALLOWED_WHISPER_MODELS, Settings
from .generation import GenerationError
from .service import Lectura
from .transcripts import ensure_ffmpeg
from .youtube import PLAYLIST_ID_RE, YOUTUBE_ID_RE, parse_youtube_url

log = logging.getLogger(__name__)


# Request models --------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    whisper_model: str | None = None

    @field_validator("whisper_model")
    @classmethod
    def _whisper(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_WHISPER_MODELS:
            raise ValueError("Unsupported whisper_model")
        return v


class QueryRequest(BaseModel):
    video_id: str | None = None
    playlist_id: str | None = None
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("video_id")
    @classmethod
    def _vid(cls, v: str | None) -> str | None:
        if v is not None and not YOUTUBE_ID_RE.match(v):
            raise ValueError("Invalid video_id")
        return v

    @field_validator("playlist_id")
    @classmethod
    def _pid(cls, v: str | None) -> str | None:
        if v is not None and not PLAYLIST_ID_RE.match(v):
            raise ValueError("Invalid playlist_id")
        return v


class ChaptersRequest(BaseModel):
    video_id: str = Field(min_length=11, max_length=11)

    @field_validator("video_id")
    @classmethod
    def _vid(cls, v: str) -> str:
        if not YOUTUBE_ID_RE.match(v):
            raise ValueError("Invalid video_id")
        return v


def _require_key(header_key: str | None, settings: Settings) -> str:
    key = (header_key or "").strip() or (settings.gemini_api_key or "")
    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing Gemini API key. Enter it in the UI header (key icon).",
        )
    return key


# App factory -------------------------------------------------------------------

def create_app(settings: Settings | None = None, service: Lectura | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    service = service or Lectura(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ensure_ffmpeg()
        try:
            log.info("Startup cleanup: %s", service.cleanup_stale())
        except Exception as e:
            log.warning("Startup cleanup failed: %s", e)
        # Warm the embedding model off the request path so the first query
        # does not pay the load.
        threading.Thread(target=lambda: service.embedder, name="warm-embedder", daemon=True).start()
        yield

    app = FastAPI(title="Lectura RAG Teaching Assistant API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def svc(request: Request) -> Lectura:
        return request.app.state.service

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        return {"version": __version__, **svc(request).health()}

    @app.get("/api/videos")
    def get_videos(request: Request) -> list[dict[str, Any]]:
        return svc(request).list_videos()

    @app.get("/api/playlists")
    def get_playlists(request: Request) -> list[dict[str, Any]]:
        return svc(request).list_playlists()

    def _ingest_video(s: Lectura, url: str, whisper_model: str | None, bg: BackgroundTasks):
        try:
            entry = s.register_video_from_url(url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.exception("Failed to register video from %s", url)
            raise HTTPException(status_code=502, detail=f"Failed to register video: {e}") from e
        bg.add_task(s.ingest_video, entry["video_id"], whisper_model)
        return {
            "kind": "video",
            "status": "processing",
            "message": f"Ingestion started for: {entry['title']}",
            "video": entry,
        }

    def _ingest_playlist(s: Lectura, url: str, whisper_model: str | None, bg: BackgroundTasks):
        try:
            info, videos = s.register_playlist_from_url(url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.exception("Failed to register playlist from %s", url)
            raise HTTPException(status_code=502, detail=f"Failed to fetch playlist: {e}") from e
        bg.add_task(s.ingest_playlist, info["playlist_id"], info["video_ids"], whisper_model)
        return {
            "kind": "playlist",
            "status": "processing",
            "message": f"Registered playlist '{info['title']}' with {len(videos)} videos. Indexing in background.",
            "playlist": info,
        }

    @app.post("/api/ingest")
    def ingest(request: Request, body: IngestRequest, background_tasks: BackgroundTasks):
        """Ingest whatever the URL points at. A watch URL that also carries a
        ``list=`` parameter is treated as a single video; use
        ``/api/ingest/playlist`` to force the playlist."""
        parsed = parse_youtube_url(body.url)
        if parsed.kind == "video":
            return _ingest_video(svc(request), body.url, body.whisper_model, background_tasks)
        if parsed.kind == "playlist":
            return _ingest_playlist(svc(request), body.url, body.whisper_model, background_tasks)
        raise HTTPException(status_code=400, detail="Not a recognisable YouTube video or playlist URL.")

    @app.post("/api/ingest/playlist")
    def ingest_playlist(request: Request, body: IngestRequest, background_tasks: BackgroundTasks):
        return _ingest_playlist(svc(request), body.url, body.whisper_model, background_tasks)

    @app.post("/api/query")
    def query(
        request: Request,
        body: QueryRequest,
        x_gemini_api_key: str | None = Header(default=None),
    ):
        s = svc(request)
        key = _require_key(x_gemini_api_key, settings)
        try:
            targets = s.resolve_targets(body.video_id, body.playlist_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            answer = s.answer(body.query, targets, key, k=body.top_k)
        except GenerationError as e:
            raise HTTPException(status_code=e.status, detail=str(e)) from e
        except Exception as e:
            log.exception("Query failed")
            raise HTTPException(status_code=500, detail=f"Failed to process query: {e}") from e
        return answer.to_dict()

    @app.post("/api/chapters")
    def chapters(
        request: Request,
        body: ChaptersRequest,
        x_gemini_api_key: str | None = Header(default=None),
    ):
        s = svc(request)
        key = _require_key(x_gemini_api_key, settings)
        try:
            completion = s.chapters(body.video_id, key)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except GenerationError as e:
            raise HTTPException(status_code=e.status, detail=str(e)) from e
        except Exception as e:
            log.exception("Chapter generation failed")
            raise HTTPException(status_code=500, detail=f"Failed to generate chapters: {e}") from e
        return {
            "chapters": completion.text,
            "usage": completion.usage.to_dict(),
            "latency_s": round(completion.latency_s, 3),
            "model": completion.model,
        }

    @app.delete("/api/videos/{video_id}")
    def delete_video(request: Request, video_id: str):
        if not YOUTUBE_ID_RE.match(video_id):
            raise HTTPException(status_code=400, detail="Invalid video_id")
        if not svc(request).delete_video(video_id):
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found in library")
        return {"status": "success", "message": f"Deleted video {video_id}."}

    @app.delete("/api/playlists/{playlist_id}")
    def delete_playlist(request: Request, playlist_id: str):
        if not PLAYLIST_ID_RE.match(playlist_id):
            raise HTTPException(status_code=400, detail="Invalid playlist_id")
        if not svc(request).delete_playlist(playlist_id):
            raise HTTPException(status_code=404, detail="Playlist not found in library")
        return {"status": "success", "message": f"Deleted playlist {playlist_id} and its videos."}

    return app
