"""Whisper transcription quality and speed against manual captions.

    python -m eval.run_asr --models tiny.en base.en small.en medium.en
    python -m eval.run_asr --models tiny.en --videos aircAruvnKk --cpu-rtf

For each video in the WER subset (manual-caption videos only):
  * download audio once into eval/data_dir/audio_eval/<id>.mp3
  * per model: transcribe on the GPU, time it (real-time factor), normalise both
    hypothesis and reference with Whisper's EnglishTextNormalizer, compute WER
    with jiwer over the whole video, count hallucination loops (a 4-gram
    repeated >= 4 times in a row), and save the transcript to
    eval/data_dir_asr_<model>/transcripts/<id>.json so the degraded-transcript
    retrieval experiment can rebuild an index from it.

Caveat printed with the table: human captions are lightly edited (fillers and
false starts removed), so WER vs captions is an upper bound on true WER;
comparisons *between* models share the reference and are fair.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from lectura.catalog import Catalog
from lectura.logging_setup import configure_logging
from lectura.transcripts import (
    SOURCE_MANUAL,
    Transcript,
    TranscriptStore,
    WhisperTranscriber,
    ensure_ffmpeg,
    whisper_source,
)
from lectura.youtube import download_audio

from .common import (
    CORPUS_DATA_DIR,
    EVAL_DIR,
    RESULTS_DIR,
    build_manifest,
    load_corpus,
    load_env,
    load_resolved_corpus,
    now_iso,
    write_json,
    write_jsonl,
)

log = logging.getLogger("eval.run_asr")

AUDIO_DIR = CORPUS_DATA_DIR / "audio_eval"


def asr_data_dir(model: str) -> Path:
    return EVAL_DIR / f"data_dir_asr_{model}"


def normalizer():
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def reference_text(t: Transcript) -> str:
    import html

    parts = []
    for ln in t.lines:
        s = html.unescape(ln.text)
        s = s.replace("\n", " ")
        s = s.replace(">>", " ")
        parts.append(s)
    return " ".join(parts)


def repeated_ngram_loops(text: str, n: int = 4, min_repeats: int = 4) -> int:
    toks = text.split()
    loops = 0
    i = 0
    while i + n * min_repeats <= len(toks):
        gram = toks[i : i + n]
        reps = 1
        while toks[i + reps * n : i + (reps + 1) * n] == gram:
            reps += 1
        if reps >= min_repeats:
            loops += 1
            i += reps * n
        else:
            i += 1
    return loops


def wer_stats(ref: str, hyp: str, norm) -> dict[str, Any]:
    import jiwer

    r, h = norm(ref), norm(hyp)
    if not r.strip():
        return {"wer": float("nan"), "ref_words": 0}
    out = jiwer.process_words(r, h)
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "hits": out.hits,
        "ref_words": len(r.split()),
        "hyp_words": len(h.split()),
    }


def clip_audio(src: Path, secs: int) -> Path:
    dst = src.with_name(f"{src.stem}_clip{secs}.mp3")
    if not dst.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(secs), "-c", "copy", str(dst)], check=True)
    return dst


def run(models: list[str], video_ids: list[str] | None, cpu_rtf: bool, clip_secs: int) -> None:
    ensure_ffmpeg()
    corpus = load_corpus()
    videos = load_resolved_corpus()
    wer_ids = video_ids or corpus.get("wer_subset") or _default_subset(videos, float(corpus.get("wer_target_hours", 4)))
    store = TranscriptStore(CORPUS_DATA_DIR / "transcripts")
    norm = normalizer()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rtf_rows: list[dict[str, Any]] = []
    gpu = WhisperTranscriber("auto")
    by_id = {v["video_id"]: v for v in videos}

    for vid in wer_ids:
        ref_t = store.load(vid)
        if ref_t is None or ref_t.source != SOURCE_MANUAL:
            log.warning("%s: reference is not manual captions (%s); skipped", vid, getattr(ref_t, "source", None))
            continue
        audio = AUDIO_DIR / f"{vid}.mp3"
        if not audio.exists():
            log.info("Downloading audio for %s", vid)
            download_audio(vid, AUDIO_DIR)
        ref = reference_text(ref_t)
        for model in models:
            try:
                t0 = time.perf_counter()
                lines = gpu.transcribe(audio, model)
                dt = time.perf_counter() - t0
            except Exception as e:  # OOM on medium.en is expected on 6 GB
                log.error("%s with %s failed: %s", vid, model, e)
                rows.append({"video_id": vid, "model": model, "error": f"{type(e).__name__}: {e}"})
                gpu.unload()
                continue
            hyp_t = Transcript(vid, lines, whisper_source(model), "en")
            # Persist for the degraded-retrieval experiment (own data dir + catalog).
            alt = asr_data_dir(model)
            TranscriptStore(alt / "transcripts").save(hyp_t)
            cat = Catalog(alt / "videos.json", alt / "playlists.json")
            meta = by_id.get(vid, {})
            cat.register_video(vid, title=meta.get("title", vid), duration=meta.get("duration", 0), author=meta.get("author") or "", thumbnail="")
            hyp = " ".join(ln.text for ln in lines)
            stats = wer_stats(ref, hyp, norm)
            row = {
                "video_id": vid,
                "course": meta.get("course"),
                "model": model,
                "device": gpu.device,
                "audio_seconds": ref_t.duration,
                "transcribe_seconds": round(dt, 1),
                "rtf": round(dt / ref_t.duration, 4) if ref_t.duration else None,
                "hallucination_loops": repeated_ngram_loops(norm(hyp)),
                **stats,
            }
            rows.append(row)
            log.info("%s %-9s WER=%.3f  RTF=%.3f  loops=%d", vid, model, row["wer"], row["rtf"] or 0, row["hallucination_loops"])
        gpu.unload()

    if cpu_rtf and wer_ids:
        cpu = WhisperTranscriber("cpu")
        first = AUDIO_DIR / f"{wer_ids[0]}.mp3"
        clip = clip_audio(first, clip_secs)
        for model in [m for m in models if not m.startswith(("medium", "large"))]:
            for dev in (gpu, cpu):
                t0 = time.perf_counter()
                dev.transcribe(clip, model)
                dt = time.perf_counter() - t0
                rtf_rows.append({"model": model, "device": dev.device, "clip_seconds": clip_secs, "seconds": round(dt, 1), "rtf": round(dt / clip_secs, 4)})
                log.info("RTF %-9s %-4s %.3f", model, dev.device, dt / clip_secs)
            dev_unload = [gpu, cpu]
            for d in dev_unload:
                d.unload()

    # Aggregate: pooled (micro) WER per model
    summary: dict[str, Any] = {}
    for model in models:
        mrows = [r for r in rows if r.get("model") == model and "wer" in r]
        if not mrows:
            summary[model] = {"n_videos": 0}
            continue
        errors = sum(r["substitutions"] + r["insertions"] + r["deletions"] for r in mrows)
        ref_words = sum(r["ref_words"] for r in mrows)
        summary[model] = {
            "n_videos": len(mrows),
            "audio_hours": round(sum(r["audio_seconds"] for r in mrows) / 3600, 2),
            "pooled_wer": errors / ref_words if ref_words else float("nan"),
            "mean_wer": sum(r["wer"] for r in mrows) / len(mrows),
            "mean_rtf": sum(r["rtf"] for r in mrows if r["rtf"]) / len(mrows),
            "hallucination_loops": sum(r["hallucination_loops"] for r in mrows),
        }
    out = RESULTS_DIR / "asr"
    write_json(out / "metrics.json", {"created": now_iso(), "models": models, "videos": wer_ids, "summary": summary, "cpu_rtf": rtf_rows,
                                      "caveat": "Reference = human captions (lightly edited); WER is an upper bound. Both sides normalised with Whisper's EnglishTextNormalizer."})
    write_jsonl(out / "per_video.jsonl", rows)
    from lectura.config import Settings

    write_json(out / "manifest.json", build_manifest(Settings(data_dir=CORPUS_DATA_DIR), sets=[], seed=0, extra={"kind": "asr"}))
    for model, s in summary.items():
        print(f"{model:10s} videos={s.get('n_videos')} pooled_WER={s.get('pooled_wer', float('nan')):.3f} mean_RTF={s.get('mean_rtf', float('nan')):.3f} loops={s.get('hallucination_loops')}")
    if shutil.which("ffmpeg") is None:
        log.warning("ffmpeg missing: CPU/GPU RTF clip step skipped")


def _default_subset(videos: list[dict[str, Any]], target_hours: float) -> list[str]:
    """One video per course first, then fill by shortest until the target."""
    chosen: list[str] = []
    seen = set()
    for v in videos:
        if v["course"] not in seen:
            chosen.append(v["video_id"])
            seen.add(v["course"])
    total = sum(v["duration"] for v in videos if v["video_id"] in chosen)
    for v in sorted(videos, key=lambda x: x["duration"]):
        if total / 3600 >= target_hours:
            break
        if v["video_id"] not in chosen:
            chosen.append(v["video_id"])
            total += v["duration"]
    return chosen


def main() -> None:
    load_env()
    configure_logging("INFO")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["tiny.en", "base.en", "small.en", "medium.en"])
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--cpu-rtf", action="store_true", help="also time tiny/base/small on CPU over a clip")
    ap.add_argument("--clip-secs", type=int, default=600)
    args = ap.parse_args()
    run(args.models, args.videos, args.cpu_rtf, args.clip_secs)


if __name__ == "__main__":
    main()
