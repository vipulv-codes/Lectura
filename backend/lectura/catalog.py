"""Library metadata: which videos and playlists exist and what state each is in.

Two small JSON files, guarded by an RLock and written atomically. This is the
only place that knows the on-disk schema of ``videos.json`` / ``playlists.json``.

Video status is explicit and written by the ingestion pipeline:

    processing   -> registered, transcript not yet available
    transcribing -> caption fetch failed, local Whisper is running (slow)
    indexed      -> transcript cached AND chunks upserted into the index
    failed       -> last attempt raised; ``error`` holds the message

The old implementation derived "ready" from the transcript file existing, which
was true *before* indexing finished and let the UI unlock unqueryable videos.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATUS_PROCESSING = "processing"
STATUS_TRANSCRIBING = "transcribing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUSES = (STATUS_PROCESSING, STATUS_TRANSCRIBING, STATUS_INDEXED, STATUS_FAILED)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        log.error("Could not read %s: %s", path.name, e)
        return {}


def atomic_write_json(path: Path, data: Any) -> None:
    """Write to a temp file in the same directory, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


class Catalog:
    def __init__(self, videos_file: Path, playlists_file: Path) -> None:
        self.videos_file = Path(videos_file)
        self.playlists_file = Path(playlists_file)
        self._lock = threading.RLock()

    # Videos ----------------------------------------------------------------

    def videos(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return _read_json(self.videos_file)

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        return self.videos().get(video_id)

    def has_video(self, video_id: str) -> bool:
        return video_id in self.videos()

    def register_video(
        self,
        video_id: str,
        *,
        title: str,
        duration: int | float,
        author: str,
        thumbnail: str,
        url: str | None = None,
    ) -> dict[str, Any]:
        """Create or reset a video entry. Re-registering resets status to
        ``processing`` and clears any previous error (this is how retry works)."""
        entry = {
            "video_id": video_id,
            "title": title,
            "duration": duration,
            "author": author,
            "thumbnail": thumbnail,
            "url": url or f"https://www.youtube.com/watch?v={video_id}",
            "status": STATUS_PROCESSING,
        }
        with self._lock:
            videos = self.videos()
            videos[video_id] = entry
            atomic_write_json(self.videos_file, videos)
        return entry

    def set_status(
        self, video_id: str, status: str, error: str | None = None, **fields: Any
    ) -> bool:
        """Update status and optionally merge extra fields (e.g.
        ``transcript_source``, ``n_chunks``). Returns False if unknown."""
        if status not in STATUSES:
            raise ValueError(f"unknown status: {status}")
        with self._lock:
            videos = self.videos()
            entry = videos.get(video_id)
            if entry is None:
                return False
            entry["status"] = status
            if error:
                entry["error"] = error
            else:
                entry.pop("error", None)
            entry.update(fields)
            atomic_write_json(self.videos_file, videos)
            return True

    def delete_video(self, video_id: str) -> bool:
        with self._lock:
            videos = self.videos()
            if video_id not in videos:
                return False
            del videos[video_id]
            atomic_write_json(self.videos_file, videos)
            return True

    # Playlists -------------------------------------------------------------

    def playlists(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return _read_json(self.playlists_file)

    def get_playlist(self, playlist_id: str) -> dict[str, Any] | None:
        return self.playlists().get(playlist_id)

    def register_playlist(self, playlist_id: str, title: str, video_ids: list[str]) -> dict[str, Any]:
        entry = {"playlist_id": playlist_id, "title": title, "video_ids": list(video_ids)}
        with self._lock:
            playlists = self.playlists()
            playlists[playlist_id] = entry
            atomic_write_json(self.playlists_file, playlists)
        return entry

    def delete_playlist(self, playlist_id: str) -> bool:
        with self._lock:
            playlists = self.playlists()
            if playlist_id not in playlists:
                return False
            del playlists[playlist_id]
            atomic_write_json(self.playlists_file, playlists)
            return True

    def remove_videos_from_playlists(self, video_ids: str | Iterable[str]) -> None:
        """Strip the given ids from every playlist so none dangles."""
        ids = {video_ids} if isinstance(video_ids, str) else set(video_ids)
        if not ids:
            return
        with self._lock:
            playlists = self.playlists()
            changed = False
            for pl in playlists.values():
                kept = [v for v in pl.get("video_ids", []) if v not in ids]
                if len(kept) != len(pl.get("video_ids", [])):
                    pl["video_ids"] = kept
                    changed = True
            if changed:
                atomic_write_json(self.playlists_file, playlists)
