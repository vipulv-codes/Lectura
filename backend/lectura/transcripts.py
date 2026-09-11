"""Transcripts: on-disk cache, YouTube caption fetch, and local Whisper fallback.

Cache format (``<data_dir>/transcripts/<video_id>.json``)::

    {"video_id": "...", "source": "captions_manual", "language": "en",
     "lines": [{"text": "...", "start": 0.0, "duration": 1.5}, ...]}

``source`` is one of ``captions_manual``, ``captions_auto``, ``whisper:<model>``
or ``unknown`` (legacy bare-list files). It matters: auto captions are
themselves ASR output and must never be used as a WER reference, and the README
should say which kind the numbers were produced on.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import atomic_write_json
from .chunking import Line, normalize_lines, transcript_duration
from .devices import resolve_device
from .youtube import YOUTUBE_ID_RE

log = logging.getLogger(__name__)

SOURCE_MANUAL = "captions_manual"
SOURCE_AUTO = "captions_auto"
SOURCE_UNKNOWN = "unknown"


def whisper_source(model_name: str) -> str:
    return f"whisper:{model_name}"


@dataclass(frozen=True)
class Transcript:
    video_id: str
    lines: list[Line]
    source: str = SOURCE_UNKNOWN
    language: str | None = None

    @property
    def duration(self) -> float:
        return transcript_duration(self.lines)

    @property
    def is_manual_captions(self) -> bool:
        return self.source == SOURCE_MANUAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source": self.source,
            "language": self.language,
            "lines": [ln.to_dict() for ln in self.lines],
        }

    @classmethod
    def from_dict(cls, video_id: str, data: Any) -> Transcript:
        # Legacy cache files were a bare list of {text,start,duration}.
        if isinstance(data, list):
            return cls(video_id, normalize_lines(Line.from_dict(d) for d in data), SOURCE_UNKNOWN, None)
        lines = normalize_lines(Line.from_dict(d) for d in data.get("lines", []))
        return cls(
            video_id=data.get("video_id") or video_id,
            lines=lines,
            source=data.get("source") or SOURCE_UNKNOWN,
            language=data.get("language"),
        )


class TranscriptStore:
    """Transcript cache directory with a small in-memory LRU (invalidated by
    file mtime so external edits are picked up)."""

    def __init__(self, directory: Path, cache_size: int = 128) -> None:
        self.directory = Path(directory)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, Transcript]] = OrderedDict()
        self._lock = threading.Lock()

    def path(self, video_id: str) -> Path:
        if not YOUTUBE_ID_RE.match(video_id):
            raise ValueError(f"invalid video id: {video_id!r}")
        return self.directory / f"{video_id}.json"

    def exists(self, video_id: str) -> bool:
        return self.path(video_id).exists()

    def load(self, video_id: str) -> Transcript | None:
        path = self.path(video_id)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            cached = self._cache.get(video_id)
            if cached and cached[0] == mtime:
                self._cache.move_to_end(video_id)
                return cached[1]
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        transcript = Transcript.from_dict(video_id, data)
        with self._lock:
            self._cache[video_id] = (mtime, transcript)
            self._cache.move_to_end(video_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return transcript

    def save(self, transcript: Transcript) -> Path:
        path = self.path(transcript.video_id)
        atomic_write_json(path, transcript.to_dict())
        with self._lock:
            self._cache.pop(transcript.video_id, None)
        return path

    def delete(self, video_id: str) -> bool:
        path = self.path(video_id)
        with self._lock:
            self._cache.pop(video_id, None)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def video_ids(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json") if YOUTUBE_ID_RE.match(p.stem))


# YouTube captions -----------------------------------------------------------

class CaptionsUnavailable(RuntimeError):
    """No usable caption track (disabled, blocked, private, or missing)."""


def list_caption_tracks(video_id: str) -> list[dict[str, Any]]:
    """Available caption tracks with their kind; used by the corpus builder to
    pick videos with human-made captions."""
    from youtube_transcript_api import YouTubeTranscriptApi

    tracks = []
    for t in YouTubeTranscriptApi().list(video_id):
        tracks.append(
            {
                "language": t.language,
                "language_code": t.language_code,
                "is_generated": bool(t.is_generated),
                "is_translatable": bool(getattr(t, "is_translatable", False)),
            }
        )
    return tracks


def fetch_caption_track(video_id: str, languages: Sequence[str] = ("en",)) -> Transcript:
    """Prefer a human-made track in ``languages``, then an auto-generated one,
    then any available track. Raises ``CaptionsUnavailable``."""
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        track = None
        for finder in ("find_manually_created_transcript", "find_generated_transcript"):
            try:
                track = getattr(transcript_list, finder)(list(languages))
                break
            except Exception:  # NoTranscriptFound and friends
                continue
        if track is None:
            track = next(iter(transcript_list), None)
        if track is None:
            raise CaptionsUnavailable(f"no caption tracks for {video_id}")
        fetched = track.fetch()
    except CaptionsUnavailable:
        raise
    except Exception as e:  # library raises a family of specific errors
        raise CaptionsUnavailable(f"{type(e).__name__}: {e}") from e

    lines = normalize_lines(Line.from_dict(s) for s in fetched)
    if not lines:
        raise CaptionsUnavailable(f"caption track for {video_id} is empty")
    source = SOURCE_AUTO if track.is_generated else SOURCE_MANUAL
    log.info("Fetched %s captions (%s) for %s: %d lines", source, track.language_code, video_id, len(lines))
    return Transcript(video_id=video_id, lines=lines, source=source, language=track.language_code)


# Whisper --------------------------------------------------------------------

def ensure_ffmpeg() -> str | None:
    """Return the ffmpeg path, adding a WinGet-installed copy to PATH if needed."""
    found = shutil.which("ffmpeg")
    if found and shutil.which("ffprobe"):
        return found
    if os.name == "nt":
        packages = Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"))
        if packages.exists():
            for root, _dirs, files in os.walk(packages):
                if "ffmpeg.exe" in files and "ffprobe.exe" in files:
                    os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                    log.info("Added FFmpeg to PATH from %s", root)
                    return os.path.join(root, "ffmpeg.exe")
    log.warning("ffmpeg/ffprobe not found on PATH; Whisper fallback will fail")
    return None


class WhisperTranscriber:
    """Keeps one Whisper model resident and swaps only when the name changes."""

    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self._model = None
        self._model_name: str | None = None
        self._lock = threading.Lock()

    def load(self, model_name: str):
        with self._lock:
            if self._model is not None and self._model_name == model_name:
                return self._model
            self.unload()
            import whisper  # heavy import (torch); keep lazy

            log.info("Loading Whisper model '%s' on %s", model_name, self.device)
            self._model = whisper.load_model(model_name, device=self.device)
            self._model_name = model_name
            return self._model

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._model_name = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(self, audio_path: Path, model_name: str, **options: Any) -> list[Line]:
        model = self.load(model_name)
        opts = {"fp16": self.device == "cuda", **options}
        result = model.transcribe(str(audio_path), **opts)
        lines = [
            Line(
                text=str(seg.get("text", "")),
                start=float(seg.get("start", 0.0)),
                duration=float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)),
            )
            for seg in result.get("segments", [])
        ]
        return normalize_lines(lines)
