from __future__ import annotations

import json

import pytest

from lectura.chunking import Line
from lectura.transcripts import (
    SOURCE_AUTO,
    SOURCE_MANUAL,
    SOURCE_UNKNOWN,
    CaptionsUnavailable,
    Transcript,
    TranscriptStore,
    fetch_caption_track,
    whisper_source,
)

VID = "aaaaaaaaaaa"


@pytest.fixture
def store(settings) -> TranscriptStore:
    return TranscriptStore(settings.transcripts_dir, cache_size=2)


def _tx(vid=VID, source=SOURCE_MANUAL):
    return Transcript(vid, [Line("b", 5.0, 1.0), Line("a", 0.0, 1.0)], source, "en")


def test_roundtrip_and_normalisation(store):
    path = store.save(_tx())
    assert path.name == f"{VID}.json"
    loaded = store.load(VID)
    assert loaded is not None
    assert [ln.text for ln in loaded.lines] == ["a", "b"]  # sorted on construction/load
    assert loaded.source == SOURCE_MANUAL and loaded.language == "en"
    assert loaded.is_manual_captions
    assert loaded.duration == 6.0
    assert store.video_ids() == [VID]
    assert store.exists(VID)


def test_missing_returns_none(store):
    assert store.load(VID) is None
    assert not store.exists(VID)
    assert not store.delete(VID)


def test_legacy_bare_list_format(store):
    store.directory.mkdir(parents=True)
    (store.directory / f"{VID}.json").write_text(
        json.dumps([{"text": "x", "start": 1, "duration": 2}]), encoding="utf-8"
    )
    t = store.load(VID)
    assert t.source == SOURCE_UNKNOWN and t.lines == [Line("x", 1.0, 2.0)]


def test_cache_invalidated_on_rewrite_and_bounded(store):
    store.save(_tx())
    first = store.load(VID)
    assert store.load(VID) is first  # cached object
    store.save(_tx(source=SOURCE_AUTO))
    assert store.load(VID).source == SOURCE_AUTO
    for i in range(3):
        store.save(_tx(vid=f"{i:011d}"))
        store.load(f"{i:011d}")
    assert len(store._cache) <= 2


def test_delete(store):
    store.save(_tx())
    assert store.delete(VID)
    assert store.load(VID) is None


def test_rejects_bad_ids(store):
    with pytest.raises(ValueError):
        store.path("../etc/passwd")
    with pytest.raises(ValueError):
        store.save(Transcript("bad id", [], SOURCE_MANUAL))


def test_whisper_source_label():
    assert whisper_source("tiny.en") == "whisper:tiny.en"


class _Track:
    def __init__(self, generated, code="en", snippets=None, fail=False):
        self.is_generated = generated
        self.language_code = code
        self.language = "English"
        self._snips = snippets if snippets is not None else [{"text": "hi", "start": 0, "duration": 1}]
        self._fail = fail

    def fetch(self):
        if self._fail:
            raise RuntimeError("blocked")
        return self._snips


class _TranscriptList:
    def __init__(self, manual=None, generated=None, others=()):
        self._manual, self._generated, self._others = manual, generated, list(others)

    def find_manually_created_transcript(self, langs):
        if self._manual is None:
            raise LookupError("none")
        return self._manual

    def find_generated_transcript(self, langs):
        if self._generated is None:
            raise LookupError("none")
        return self._generated

    def __iter__(self):
        return iter([t for t in (self._manual, self._generated, *self._others) if t])


def _patch_api(monkeypatch, transcript_list):
    class _Api:
        def list(self, video_id):
            return transcript_list

    import youtube_transcript_api

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", _Api)


def test_fetch_prefers_manual(monkeypatch):
    _patch_api(monkeypatch, _TranscriptList(manual=_Track(False), generated=_Track(True)))
    t = fetch_caption_track(VID)
    assert t.source == SOURCE_MANUAL and t.lines == [Line("hi", 0.0, 1.0)]


def test_fetch_falls_back_to_generated_then_any(monkeypatch):
    _patch_api(monkeypatch, _TranscriptList(generated=_Track(True)))
    assert fetch_caption_track(VID).source == SOURCE_AUTO
    _patch_api(monkeypatch, _TranscriptList(others=[_Track(False, code="de")]))
    t = fetch_caption_track(VID)
    assert t.source == SOURCE_MANUAL and t.language == "de"


def test_fetch_raises_when_nothing(monkeypatch):
    _patch_api(monkeypatch, _TranscriptList())
    with pytest.raises(CaptionsUnavailable):
        fetch_caption_track(VID)
    _patch_api(monkeypatch, _TranscriptList(manual=_Track(False, fail=True)))
    with pytest.raises(CaptionsUnavailable, match="RuntimeError"):
        fetch_caption_track(VID)
    _patch_api(monkeypatch, _TranscriptList(manual=_Track(False, snippets=[])))
    with pytest.raises(CaptionsUnavailable, match="empty"):
        fetch_caption_track(VID)
