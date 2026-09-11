"""Device selection shared by the embedder, reranker and Whisper."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_device(device: str = "auto") -> str:
    """``auto`` -> ``cuda`` if torch sees a GPU, else ``cpu``. Explicit values
    pass through (``cuda`` falls back to ``cpu`` with a warning if unavailable)."""
    if device == "cpu":
        return "cpu"
    try:
        import torch  # local import: torch is slow to load and optional for tests
    except ImportError:
        if device == "cuda":
            log.warning("torch not installed; falling back to cpu")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if device == "cuda":
        log.warning("CUDA requested but not available; falling back to cpu")
    return "cpu"
