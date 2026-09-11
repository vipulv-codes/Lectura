"""One place to configure logging for the server and the eval CLIs."""

from __future__ import annotations

import logging

_NOISY = ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3", "filelock", "huggingface_hub")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
