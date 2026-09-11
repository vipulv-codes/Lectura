"""The application object: wires catalog, transcripts, index, retrieval and
generation together. The HTTP layer and the evaluation harness both talk to
this class, so they exercise exactly the same code path.

Heavy resources (embedding model, Chroma client, Whisper) are created lazily
and once.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .catalog import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PROCESSING,
    STATUS_TRANSCRIBING,
    Catalog,
)
from .chunking import chunk_transcript
from .config import Settings
from .embeddings import Embedder, get_embedder
from .generation import Answer, Completion, GeminiGenerator, build_prompt, no_context_answer
from .index import ChromaIndex
from .retrieval import RetrievalPipeline, RetrievalResult
from .transcripts import (
    CaptionsUnavailable,
    Transcript,
    TranscriptStore,
    WhisperTranscriber,
    fetch_caption_track,
    whisper_source,
)
from .youtube import (
    download_audio,
    fetch_playlist_details,
    fetch_video_details,
    parse_youtube_url,
    thumbnail_url,
    watch_url,
)

log = logging.getLogger(__name__)

GeneratorFactory = Callable[[str], GeminiGenerator]


class Lectura:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        generator_factory: GeneratorFactory | None = None,
    ) -> None:
        self.settings = settings
        self.catalog = Catalog(settings.videos_file, settings.playlists_file)
        self.transcripts = TranscriptStore(settings.transcripts_dir)
        self._embedder = embedder
        self._index: ChromaIndex | None = None
        self._pipeline: RetrievalPipeline | None = None
        self._whisper: WhisperTranscriber | None = None
        self._generator_factory = generator_factory or (
            lambda key: GeminiGenerator(key, settings.generation)
        )
        self._lock = threading.RLock()

    # Lazy resources ----------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        with self._lock:
            if self._embedder is None:
                cfg = self.settings.index
                self._embedder = get_embedder(
                    cfg.embedding_model, cfg.doc_prefix, cfg.query_prefix, self.settings.device
                )
            return self._embedder

    @property
    def index(self) -> ChromaIndex:
        with self._lock:
            if self._index is None:
                self._index = ChromaIndex(self.settings.chroma_dir, self.settings.index, self.embedder)
            return self._index

    @property
    def pipeline(self) -> RetrievalPipeline:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = RetrievalPipeline(
                    self.index, self.settings.retrieval, self.settings.device
                )
            return self._pipeline

    @property
    def whisper(self) -> WhisperTranscriber:
        with self._lock:
            if self._whisper is None:
                self._whisper = WhisperTranscriber(self.settings.device)
            return self._whisper

    # Library views -----------------------------------------------------------

    def list_videos(self) -> list[dict[str, Any]]:
        return [self._public_video(v) for v in self.catalog.videos().values()]

    def list_playlists(self) -> list[dict[str, Any]]:
        videos = self.catalog.videos()
        out = []
        for pid, pl in self.catalog.playlists().items():
            nested = []
            for vid in pl.get("video_ids", []):
                if vid in videos:
                    nested.append(self._public_video(videos[vid]))
                else:
                    nested.append(
                        {
                            "video_id": vid,
                            "title": f"Queued Video ({vid})",
                            "duration": 0,
                            "author": "YouTube",
                            "thumbnail": thumbnail_url(vid),
                            "url": watch_url(vid),
                            "status": STATUS_PROCESSING,
                        }
                    )
            out.append({"playlist_id": pid, "title": pl.get("title", f"Playlist {pid}"), "videos": nested})
        return out

    @staticmethod
    def _public_video(entry: dict[str, Any]) -> dict[str, Any]:
        return {**entry, "status": entry.get("status", STATUS_PROCESSING)}

    def health(self) -> dict[str, Any]:
        videos = self.catalog.videos()
        counts = {s: 0 for s in (STATUS_PROCESSING, STATUS_TRANSCRIBING, STATUS_INDEXED, STATUS_FAILED)}
        for v in videos.values():
            counts[v.get("status", STATUS_PROCESSING)] = counts.get(v.get("status", STATUS_PROCESSING), 0) + 1
        return {
            "status": "ok",
            "videos": counts,
            "playlists": len(self.catalog.playlists()),
            "config": self.settings.to_dict(),
        }

    # Ingestion ---------------------------------------------------------------

    def register_video_from_url(self, url: str) -> dict[str, Any]:
        parsed = parse_youtube_url(url)
        if not parsed.video_id:
            raise ValueError(f"Could not extract a valid YouTube video ID from URL: {url}")
        return self.register_video(parsed.video_id)

    def register_video(self, video_id: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        details = details or fetch_video_details(video_id)
        return self.catalog.register_video(
            video_id,
            title=details["title"],
            duration=details.get("duration") or 0,
            author=details.get("author") or "YouTube",
            thumbnail=details.get("thumbnail") or thumbnail_url(video_id),
            url=details.get("url"),
        )

    def register_playlist_from_url(self, url: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        parsed = parse_youtube_url(url)
        if not parsed.playlist_id:
            raise ValueError(f"Could not extract a valid playlist ID from URL: {url}")
        details = fetch_playlist_details(parsed.playlist_id)
        videos = details["videos"]
        if len(videos) > self.settings.max_playlist_videos:
            raise ValueError(
                f"Playlist has {len(videos)} videos, exceeding the limit of "
                f"{self.settings.max_playlist_videos}."
            )
        for v in videos:
            self.register_video(v["video_id"], v)
        info = self.catalog.register_playlist(
            parsed.playlist_id, details["title"], [v["video_id"] for v in videos]
        )
        return info, videos

    def _obtain_transcript(self, video_id: str, whisper_model: str) -> Transcript:
        cached = self.transcripts.load(video_id)
        if cached is not None and cached.lines:
            log.info("Using cached transcript for %s (%s)", video_id, cached.source)
            return cached
        try:
            transcript = fetch_caption_track(video_id)
        except CaptionsUnavailable as e:
            log.info("No captions for %s (%s); falling back to Whisper %s", video_id, e, whisper_model)
            transcript = self._transcribe_with_whisper(video_id, whisper_model)
        if not self.catalog.has_video(video_id):
            raise _Deleted(video_id)
        self.transcripts.save(transcript)
        return transcript

    def _transcribe_with_whisper(self, video_id: str, whisper_model: str) -> Transcript:
        self.catalog.set_status(video_id, STATUS_TRANSCRIBING)
        audio: Path | None = None
        try:
            audio = download_audio(video_id, self.settings.audio_dir)
            if not self.catalog.has_video(video_id):
                raise _Deleted(video_id)
            lines = self.whisper.transcribe(audio, whisper_model)
            return Transcript(video_id, lines, whisper_source(whisper_model), "en")
        finally:
            if audio is not None:
                for p in self.settings.audio_dir.glob(f"{video_id}.*"):
                    try:
                        p.unlink()
                    except OSError as e:
                        log.warning("Could not remove %s: %s", p, e)

    def ingest_video(self, video_id: str, whisper_model: str | None = None) -> dict[str, Any] | None:
        """Background-worker body: transcript (captions, else Whisper) -> chunks
        -> index -> ``indexed``. Any failure is recorded as ``failed`` with the
        message so the UI can offer a retry instead of spinning forever."""
        whisper_model = whisper_model or self.settings.whisper_model
        if not self.catalog.has_video(video_id):
            log.info("Video %s was deleted before ingestion started; skipping", video_id)
            return None
        try:
            transcript = self._obtain_transcript(video_id, whisper_model)
            chunks = chunk_transcript(transcript.lines, self.settings.index.child_secs, video_id)
            if not self.catalog.has_video(video_id):
                raise _Deleted(video_id)
            self.index.delete_video(video_id)
            n = self.index.add_chunks(chunks)
            self.pipeline.invalidate()
            self.catalog.set_status(
                video_id,
                STATUS_INDEXED,
                transcript_source=transcript.source,
                transcript_language=transcript.language,
                n_lines=len(transcript.lines),
                n_chunks=n,
                collection=self.index.name,
            )
            log.info("Indexed %s: %d chunks (%s)", video_id, n, transcript.source)
        except _Deleted:
            log.info("Video %s was deleted during ingestion; aborting", video_id)
            return None
        except Exception as e:
            log.exception("Ingestion failed for %s", video_id)
            self.catalog.set_status(video_id, STATUS_FAILED, error=f"{type(e).__name__}: {e}")
        return self.catalog.get_video(video_id)

    def ingest_playlist(
        self, playlist_id: str, video_ids: Sequence[str], whisper_model: str | None = None
    ) -> None:
        for i, vid in enumerate(video_ids, 1):
            if self.catalog.get_playlist(playlist_id) is None:
                log.info("Playlist %s deleted; stopping after %d videos", playlist_id, i - 1)
                return
            log.info("[%d/%d] playlist %s: %s", i, len(video_ids), playlist_id, vid)
            self.ingest_video(vid, whisper_model)

    # Deletion ----------------------------------------------------------------

    def delete_video(self, video_id: str) -> bool:
        existed = self.catalog.delete_video(video_id)
        try:
            self.index.delete_video(video_id)
            self.pipeline.invalidate()
        except Exception as e:  # keep going: metadata is already gone
            log.warning("Could not delete embeddings for %s: %s", video_id, e)
        self.transcripts.delete(video_id)
        for p in self.settings.audio_dir.glob(f"{video_id}.*"):
            with contextlib.suppress(OSError):
                p.unlink()
        self.catalog.remove_videos_from_playlists(video_id)
        return existed

    def delete_playlist(self, playlist_id: str) -> bool:
        pl = self.catalog.get_playlist(playlist_id)
        if pl is None:
            return False
        for vid in pl.get("video_ids", []):
            self.delete_video(vid)
        self.catalog.delete_playlist(playlist_id)
        return True

    def cleanup_stale(self) -> dict[str, int]:
        """Startup hygiene: leftover audio, transcripts and embeddings without a
        catalog entry, and catalog entries stuck in a transient state."""
        removed = {"audio": 0, "transcripts": 0, "embeddings": 0, "requeued": 0}
        if self.settings.audio_dir.exists():
            for p in self.settings.audio_dir.iterdir():
                try:
                    p.unlink()
                    removed["audio"] += 1
                except OSError:
                    pass
        known = set(self.catalog.videos())
        for vid in self.transcripts.video_ids():
            if vid not in known:
                self.transcripts.delete(vid)
                removed["transcripts"] += 1
        try:
            for vid in self.index.video_ids() - known:
                self.index.delete_video(vid)
                removed["embeddings"] += 1
        except Exception as e:
            log.warning("Orphan embedding cleanup skipped: %s", e)
        # A process restart kills background tasks; anything still marked as
        # in-flight will never finish, so surface it as failed-with-retry.
        for vid, entry in self.catalog.videos().items():
            if entry.get("status", STATUS_PROCESSING) in (STATUS_PROCESSING, STATUS_TRANSCRIBING):
                self.catalog.set_status(vid, STATUS_FAILED, error="Interrupted by server restart; retry.")
                removed["requeued"] += 1

        # Changing the embedding model or chunk size derives a new collection
        # name, so a library indexed under the old settings would still say
        # "indexed" while its new collection is empty and every query returns
        # nothing. Re-embed those from the cached transcripts: no network, no
        # re-transcription, and the alternative is a silently broken library.
        removed["reindexed"] = self._reindex_missing()
        return removed

    def _reindex_missing(self) -> int:
        count = 0
        for vid, entry in self.catalog.videos().items():
            if entry.get("status") != STATUS_INDEXED:
                continue
            try:
                if self.index.chunk_count(vid) > 0:
                    continue
            except Exception as e:
                log.warning("Could not check the index for %s: %s", vid, e)
                return count
            transcript = self.transcripts.load(vid)
            if transcript is None or not transcript.lines:
                self.catalog.set_status(vid, STATUS_FAILED, error="Indexed but no transcript cached; retry.")
                continue
            log.info("Re-indexing %s into %s (config changed)", vid, self.index.name)
            chunks = chunk_transcript(transcript.lines, self.settings.index.child_secs, vid)
            n = self.index.add_chunks(chunks)
            self.catalog.set_status(
                vid, STATUS_INDEXED, transcript_source=transcript.source,
                transcript_language=transcript.language, n_lines=len(transcript.lines),
                n_chunks=n, collection=self.index.name,
            )
            count += 1
        if count:
            self.pipeline.invalidate()
        return count

    # Query -------------------------------------------------------------------

    def resolve_targets(
        self, video_id: str | None = None, playlist_id: str | None = None
    ) -> list[str]:
        if playlist_id:
            pl = self.catalog.get_playlist(playlist_id)
            if pl is None:
                raise LookupError("Playlist not found")
            return list(pl.get("video_ids", []))
        if video_id:
            if not self.catalog.has_video(video_id):
                raise LookupError("Video not found")
            return [video_id]
        raise ValueError("Must provide either video_id or playlist_id")

    def retrieve(self, query: str, video_ids: Sequence[str] | None, k: int | None = None) -> RetrievalResult:
        if k is None and video_ids is not None and len(video_ids) > 1:
            k = self.settings.retrieval.top_k_playlist
        return self.pipeline.retrieve(query, video_ids, k)

    def titles(self, video_ids: Sequence[str]) -> dict[str, str]:
        videos = self.catalog.videos()
        return {v: videos[v]["title"] for v in video_ids if v in videos}

    def answer(
        self,
        query: str,
        video_ids: Sequence[str] | None,
        api_key: str | None,
        k: int | None = None,
        generator: GeminiGenerator | None = None,
    ) -> Answer:
        result = self.retrieve(query, video_ids, k)
        retrieval_info = {
            "covered": result.covered,
            "best_score": result.gate_score,
            "n_candidates": len(result.candidates),
            "timings": {s: round(t, 4) for s, t in result.timings.items()},
        }
        if not result.hits or not result.covered:
            ans = no_context_answer(query)
            ans.extra["retrieval"] = retrieval_info
            return ans
        prompt = build_prompt(
            query,
            result.hits,
            self.transcripts,
            self.titles([h.chunk.video_id for h in result.hits]),
            self.settings.retrieval.parent_secs,
            self.settings.generation.context_order,
        )
        gen = generator or self._generator_factory(api_key or self.settings.gemini_api_key or "")
        ans = gen.answer(prompt)
        ans.extra["retrieval"] = retrieval_info
        return ans

    def chapters(self, video_id: str, api_key: str | None, generator: GeminiGenerator | None = None) -> Completion:
        transcript = self.transcripts.load(video_id)
        if transcript is None:
            raise LookupError("Video transcript not found. Please ingest the video first.")
        title = self.titles([video_id]).get(video_id, f"Video {video_id}")
        gen = generator or self._generator_factory(api_key or self.settings.gemini_api_key or "")
        return gen.chapters(transcript, title)


class _Deleted(Exception):
    """Raised internally when a video disappears from the catalog mid-ingest."""
