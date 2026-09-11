"""YouTube URL parsing and yt-dlp access (metadata, playlists, audio download).

IDs are validated against strict patterns before they are ever used as cache
file names or request targets, so a crafted URL cannot inject path segments.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
PLAYLIST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,64}$")
YOUTUBE_HOSTS = frozenset(
    {
        "youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com",
        "music.youtube.com", "www.youtube-nocookie.com",
    }
)


@dataclass(frozen=True)
class ParsedUrl:
    video_id: str | None
    playlist_id: str | None

    @property
    def kind(self) -> Literal["video", "playlist"] | None:
        # A watch URL inside a playlist (v=...&list=...) is a *video* request:
        # that is what YouTube hands you when you click a video in a playlist,
        # and treating it as a playlist ingests up to hundreds of videos.
        if self.video_id:
            return "video"
        if self.playlist_id:
            return "playlist"
        return None


def parse_youtube_url(url: str) -> ParsedUrl:
    """Extract validated video and/or playlist ids. Non-YouTube hosts yield
    nothing. A bare 11-character id is accepted as a video id."""
    url = url.strip()
    if YOUTUBE_ID_RE.match(url):
        return ParsedUrl(video_id=url, playlist_id=None)
    if "://" not in url:
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        return ParsedUrl(None, None)

    query = urllib.parse.parse_qs(parsed.query)
    video_candidate: str | None = None
    if host == "youtu.be":
        video_candidate = parsed.path.strip("/").split("/")[0] or None
    else:
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path == "/watch":
            video_candidate = query.get("v", [None])[0]
        elif len(parts) >= 2 and parts[0] in ("embed", "v", "shorts", "live"):
            video_candidate = parts[1]

    playlist_candidate = query.get("list", [None])[0]
    video_id = video_candidate if video_candidate and YOUTUBE_ID_RE.match(video_candidate) else None
    playlist_id = (
        playlist_candidate
        if playlist_candidate and PLAYLIST_ID_RE.match(playlist_candidate)
        else None
    )
    return ParsedUrl(video_id=video_id, playlist_id=playlist_id)


def extract_video_id(url: str) -> str:
    parsed = parse_youtube_url(url)
    if not parsed.video_id:
        raise ValueError(f"Could not extract a valid YouTube video ID from URL: {url}")
    return parsed.video_id


def extract_playlist_id(url: str) -> str:
    parsed = parse_youtube_url(url)
    if not parsed.playlist_id:
        raise ValueError(f"Could not extract a valid playlist ID from URL: {url}")
    return parsed.playlist_id


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def thumbnail_url(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


# yt-dlp ---------------------------------------------------------------------

def _ydl(opts: dict[str, Any]):
    import yt_dlp  # slow import; keep it out of module load

    base = {"quiet": True, "no_warnings": True}
    return yt_dlp.YoutubeDL({**base, **opts})


def fetch_video_details(video_id: str) -> dict[str, Any]:
    """Title/duration/author without downloading media. Falls back to
    placeholders on any failure so the library entry can still be created."""
    url = watch_url(video_id)
    fallback = {
        "video_id": video_id,
        "title": f"Video {video_id}",
        "duration": 0,
        "author": "YouTube",
        "thumbnail": thumbnail_url(video_id),
        "url": url,
    }
    try:
        with _ydl({"skip_download": True, "extract_flat": True}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as e:  # yt-dlp raises many unrelated exception types
        log.warning("yt-dlp metadata fetch failed for %s: %s", video_id, e)
        return fallback
    return {
        **fallback,
        "title": info.get("title") or fallback["title"],
        "duration": info.get("duration") or 0,
        "author": info.get("uploader") or info.get("channel") or "YouTube",
    }


def fetch_playlist_details(playlist_id: str) -> dict[str, Any]:
    """Playlist title and flat list of entries (no media download)."""
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    with _ydl({"extract_flat": True}) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    videos = []
    for entry in info.get("entries") or []:
        if not entry:
            continue
        vid = entry.get("id")
        if not vid or not YOUTUBE_ID_RE.match(vid):
            continue
        videos.append(
            {
                "video_id": vid,
                "title": entry.get("title") or f"Video {vid}",
                "duration": entry.get("duration") or 0,
                "author": entry.get("uploader") or info.get("uploader") or "YouTube",
                "thumbnail": thumbnail_url(vid),
                "url": watch_url(vid),
            }
        )
    return {
        "playlist_id": playlist_id,
        "title": info.get("title") or f"Playlist ({playlist_id})",
        "videos": videos,
    }


def download_audio(video_id: str, audio_dir: Path) -> Path:
    """Download the best audio track as MP3 into ``audio_dir``."""
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(audio_dir / f"{video_id}.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "outtmpl": out_tmpl,
    }
    with _ydl(opts) as ydl:
        ydl.download([watch_url(video_id)])
    mp3 = audio_dir / f"{video_id}.mp3"
    if not mp3.exists():
        raise FileNotFoundError(f"Audio was not produced: {mp3}")
    return mp3
