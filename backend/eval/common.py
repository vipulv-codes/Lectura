"""Shared plumbing for the eval CLIs: paths, config loading, manifests, and a
cached + throttled Gemini client so re-scoring never re-spends tokens."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lectura.config import Settings
from lectura.generation import Completion, GeminiGenerator, GenerationError, Usage
from lectura.service import Lectura

from .gold import GoldItem, dataset_sha256, load_jsonl

log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"            # committed datasets
CORPUS_DATA_DIR = EVAL_DIR / "data_dir"  # isolated LECTURA_DATA_DIR (gitignored)
CACHE_DIR = EVAL_DIR / "cache"
RESULTS_DIR = EVAL_DIR / "results"
CONFIGS_DIR = EVAL_DIR / "configs"
CORPUS_FILE = EVAL_DIR / "corpus.yaml"
CORPUS_RESOLVED = DATA_DIR / "corpus_resolved.json"
PRICING_FILE = EVAL_DIR / "pricing.yaml"

GOLD_FILES = {
    "human": DATA_DIR / "human.jsonl",
    "synthetic": DATA_DIR / "synthetic.jsonl",
    "unanswerable": DATA_DIR / "unanswerable.jsonl",
}


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# JSON helpers ---------------------------------------------------------------

def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    if isinstance(o, set | frozenset):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


# Corpus ---------------------------------------------------------------------------

def load_corpus() -> dict[str, Any]:
    with CORPUS_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_resolved_corpus() -> list[dict[str, Any]]:
    if not CORPUS_RESOLVED.exists():
        raise SystemExit(f"{CORPUS_RESOLVED} missing: run `python -m eval.build_corpus probe` first")
    return read_json(CORPUS_RESOLVED)["videos"]


def corpus_video_ids() -> list[str]:
    return [v["video_id"] for v in load_resolved_corpus()]


# Configs ---------------------------------------------------------------------------

def load_config(name_or_path: str) -> tuple[str, Settings, dict[str, Any]]:
    """A config YAML holds ``name`` plus optional ``index``/``retrieval``/
    ``generation`` sections overlaying the defaults. Data dir is always the
    isolated eval one."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    name = raw.get("name") or path.stem
    overlay = {k: v for k, v in raw.items() if k in ("index", "retrieval", "generation", "device", "whisper_model")}
    base = Settings(data_dir=CORPUS_DATA_DIR, gemini_api_key=os.getenv("GEMINI_API_KEY"))
    return name, Settings.from_dict(overlay, base=base), raw


def settings_from_dict(overlay: dict[str, Any], name: str = "inline", data_dir: Path | None = None) -> Settings:
    base = Settings(data_dir=data_dir or CORPUS_DATA_DIR, gemini_api_key=os.getenv("GEMINI_API_KEY"))
    return Settings.from_dict(overlay, base=base)


def make_service(settings: Settings) -> Lectura:
    return Lectura(settings)


# Gold ------------------------------------------------------------------------------

def load_gold(sets: list[str]) -> list[GoldItem]:
    items: list[GoldItem] = []
    for s in sets:
        path = GOLD_FILES[s]
        if not path.exists():
            log.warning("Gold set %s missing at %s; skipping", s, path)
            continue
        items.extend(load_jsonl(path))
    return items


def gold_hashes(sets: list[str]) -> dict[str, str]:
    return {s: dataset_sha256(GOLD_FILES[s]) for s in sets if GOLD_FILES[s].exists()}


# Manifest ---------------------------------------------------------------------------

def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR, text=True).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in ("chromadb", "sentence-transformers", "transformers", "torch", "google-genai",
                 "openai-whisper", "youtube-transcript-api", "yt-dlp", "rank-bm25", "jiwer", "numpy"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "missing"
    return out


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def build_manifest(settings: Settings, *, sets: list[str], seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "created": now_iso(),
        "git_sha": git_sha(),
        "config": settings.to_dict(),
        "seed": seed,
        "datasets": gold_hashes(sets),
        "corpus": dataset_sha256(CORPUS_RESOLVED) if CORPUS_RESOLVED.exists() else None,
        "packages": package_versions(),
        "device": device_info(),
        **(extra or {}),
    }


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# Gemini with cache + throttle ------------------------------------------------------------

class CachedGemini:
    """Wraps ``GeminiGenerator.complete`` with a content-addressed disk cache
    (sha256 of model + generation config + system + user) and a requests-per-
    minute limiter with retry on 429. Re-running a scorer is free."""

    def __init__(self, generator: GeminiGenerator, cache_dir: Path = CACHE_DIR / "gemini", rpm: float | None = None, max_retries: int = 6) -> None:
        self.generator = generator
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.max_retries = max_retries
        self._last_call = 0.0
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "retries": 0}

    def _key(self, system: str, user: str) -> str:
        cfg = json.dumps(asdict(self.generator.cfg), sort_keys=True)
        return hashlib.sha256(f"{cfg}\n{system}\n{user}".encode()).hexdigest()

    def complete(self, system: str, user: str, tag: str = "") -> Completion:
        key = self._key(system, user)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            d = read_json(path)
            self.stats["hits"] += 1
            return Completion(d["text"], Usage(**d["usage"]), d["latency_s"], d["model"], d.get("model_version"))
        self.stats["misses"] += 1
        attempt = 0
        while True:
            with self._lock:
                wait = self.min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.monotonic()
            try:
                completion = self.generator.complete(system, user)
                break
            except GenerationError as e:
                if e.status == 429 and attempt < self.max_retries:
                    attempt += 1
                    self.stats["retries"] += 1
                    backoff = min(60.0, 2.0**attempt)
                    log.warning("429 from Gemini; retry %d in %.0fs (%s)", attempt, backoff, tag)
                    time.sleep(backoff)
                    continue
                raise
        write_json(
            path,
            {
                "text": completion.text,
                "usage": completion.usage.to_dict(),
                "latency_s": completion.latency_s,
                "model": completion.model,
                "model_version": completion.model_version,
                "tag": tag,
                "created": now_iso(),
                "system_sha": hashlib.sha256(system.encode()).hexdigest(),
                "user_sha": hashlib.sha256(user.encode()).hexdigest(),
            },
        )
        return completion


def load_pricing() -> dict[str, Any]:
    if not PRICING_FILE.exists():
        return {}
    with PRICING_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def price_for(model: str, pricing: dict[str, Any]) -> dict[str, float]:
    models = pricing.get("models", {})
    entry = models.get(model) or {}
    return {"input_per_m": float(entry.get("input_per_m", 0.0)), "output_per_m": float(entry.get("output_per_m", 0.0))}
