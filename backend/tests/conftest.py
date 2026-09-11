from __future__ import annotations

import pytest

from lectura.catalog import Catalog
from lectura.chunking import Line
from lectura.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def catalog(settings) -> Catalog:
    return Catalog(settings.videos_file, settings.playlists_file)


@pytest.fixture
def lines() -> list[Line]:
    """A 100-second transcript with one line every 10 s (durations 8 s)."""
    return [Line(f"line {i}", float(i * 10), 8.0) for i in range(10)]
