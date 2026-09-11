"""Resolve, verify and ingest the benchmark corpus.

    python -m eval.build_corpus probe    # resolve playlists, check captions, write data/corpus_resolved.json
    python -m eval.build_corpus ingest   # fetch transcripts + build the baseline index in eval/data_dir
    python -m eval.build_corpus status   # what is in the isolated data dir

`probe` never writes to the data dir, so the resolved list can be reviewed
before anything is downloaded.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from lectura.logging_setup import configure_logging
from lectura.transcripts import list_caption_tracks
from lectura.youtube import fetch_playlist_details

from .common import (
    CORPUS_DATA_DIR,
    CORPUS_RESOLVED,
    load_corpus,
    load_env,
    load_resolved_corpus,
    make_service,
    now_iso,
    read_json,
    settings_from_dict,
    write_json,
)

log = logging.getLogger("eval.build_corpus")


def probe() -> list[dict[str, Any]]:
    corpus = load_corpus()
    resolved: list[dict[str, Any]] = []
    for src in corpus["sources"]:
        details = fetch_playlist_details(src["playlist"])
        log.info("Playlist %s: '%s' (%d entries)", src["playlist"], details["title"], len(details["videos"]))
        kept = 0
        for entry in details["videos"]:
            if kept >= int(src.get("max_videos", 999)):
                break
            vid = entry["video_id"]
            try:
                tracks = list_caption_tracks(vid)
            except Exception as e:
                log.warning("  %s: caption listing failed (%s); skipped", vid, e)
                continue
            en = [t for t in tracks if t["language_code"].lower().startswith("en")]
            manual = [t for t in en if not t["is_generated"]]
            if not manual:
                log.warning("  %s: no manual English captions (%d tracks); skipped", vid, len(tracks))
                continue
            kept += 1
            resolved.append(
                {
                    "video_id": vid,
                    "title": entry["title"],
                    "duration": int(entry.get("duration") or 0),
                    "author": entry.get("author"),
                    "course": src["course"],
                    "playlist": src["playlist"],
                    "playlist_title": details["title"],
                    "captions": {"manual_en": len(manual), "auto_en": len(en) - len(manual), "tracks": len(tracks)},
                }
            )
            log.info("  + %s  %6ds  %s", vid, entry.get("duration") or 0, entry["title"][:70])
    total = sum(v["duration"] for v in resolved)
    write_json(CORPUS_RESOLVED, {"name": corpus["name"], "resolved": now_iso(), "videos": resolved,
                                 "total_seconds": total, "n_videos": len(resolved)})
    log.info("Resolved %d videos, %.1f h of audio -> %s", len(resolved), total / 3600, CORPUS_RESOLVED)
    return resolved


def ingest(config_overlay: dict[str, Any] | None = None) -> None:
    videos = load_resolved_corpus()
    settings = settings_from_dict(config_overlay or {})
    svc = make_service(settings)
    log.info("Ingesting %d videos into %s (collection %s)", len(videos), CORPUS_DATA_DIR, settings.index.collection_name())
    ingest_log: list[dict[str, Any]] = []
    for i, v in enumerate(videos, 1):
        vid = v["video_id"]
        entry = svc.catalog.get_video(vid)
        if entry and entry.get("status") == "indexed" and svc.index.chunk_count(vid) > 0:
            log.info("[%d/%d] %s already indexed (%s chunks)", i, len(videos), vid, entry.get("n_chunks"))
            continue
        svc.register_video(vid, {**v, "thumbnail": None, "url": None})
        t0 = time.perf_counter()
        had_transcript = svc.transcripts.exists(vid)
        entry = svc.ingest_video(vid) or {}
        dt = time.perf_counter() - t0
        rec = {
            "video_id": vid,
            "status": entry.get("status"),
            "transcript_source": entry.get("transcript_source"),
            "n_lines": entry.get("n_lines"),
            "n_chunks": entry.get("n_chunks"),
            "duration": v["duration"],
            "seconds": round(dt, 2),
            "transcript_cached": had_transcript,
            "error": entry.get("error"),
        }
        ingest_log.append(rec)
        log.info("[%d/%d] %s -> %s in %.1fs (%s chunks, %s)", i, len(videos), vid, rec["status"], dt, rec["n_chunks"], rec["transcript_source"])
    if ingest_log:
        write_json(CORPUS_DATA_DIR / "ingest_log.json", {"created": now_iso(), "config": settings.to_dict(), "videos": ingest_log})
    failed = [r for r in ingest_log if r["status"] != "indexed"]
    if failed:
        log.error("%d videos failed: %s", len(failed), [r["video_id"] for r in failed])


def status() -> None:
    settings = settings_from_dict({})
    svc = make_service(settings)
    for vid, entry in svc.catalog.videos().items():
        print(f"{vid}  {entry.get('status'):12s}  {entry.get('transcript_source', '-'):16s}  chunks={entry.get('n_chunks', '-'):>4}  {entry.get('title', '')[:60]}")
    if CORPUS_RESOLVED.exists():
        r = read_json(CORPUS_RESOLVED)
        print(f"corpus: {r['n_videos']} videos, {r['total_seconds'] / 3600:.1f} h")


def main() -> None:
    load_env()
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probe", "ingest", "status"])
    args = ap.parse_args()
    {"probe": probe, "ingest": ingest, "status": status}[args.command]()


if __name__ == "__main__":
    main()
