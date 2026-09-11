"""Entry point: ``python main.py`` (or ``uvicorn main:app``)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from lectura.api import create_app  # noqa: E402
from lectura.config import Settings  # noqa: E402
from lectura.logging_setup import configure_logging  # noqa: E402

settings = Settings.from_env()
configure_logging(settings.log_level)
app = create_app(settings)

if __name__ == "__main__":
    import uvicorn

    # Loopback by default: this API has no auth. Set LECTURA_HOST=0.0.0.0 to
    # expose it on the LAN deliberately. LECTURA_RELOAD=1 enables autoreload.
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("LECTURA_RELOAD") == "1",
        log_level=settings.log_level.lower(),
    )
