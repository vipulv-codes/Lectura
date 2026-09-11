from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lectura import service as service_mod
from lectura.api import create_app
from lectura.chunking import Line
from lectura.embeddings import HashEmbedder
from lectura.generation import Answer, Completion, GenerationError, Usage
from lectura.service import Lectura
from lectura.transcripts import SOURCE_MANUAL, CaptionsUnavailable, Transcript, WhisperTranscriber

VID = "aaaaaaaaaaa"
VID2 = "bbbbbbbbbbb"
PL = "PLxxxxxxxxxxxxxxxx"
KEY = {"X-Gemini-API-Key": "k"}


def _lines(prefix="x"):
    return [Line(f"{prefix} sentence {i} about topic {i}", i * 10.0, 8.0) for i in range(12)]


class FakeGenerator:
    def __init__(self, fail: GenerationError | None = None):
        self.fail = fail
        self.prompts = []

    def answer(self, prompt):
        if self.fail:
            raise self.fail
        self.prompts.append(prompt)
        first = sorted(prompt.valid_timestamps)[0]
        return Answer(
            text=f"Answer citing [{first}].", sources=prompt.sources, abstained=False,
            citations=((first, 0.0),), invalid_citations=(), usage=Usage(1, 1, 0, 2),
            latency_s=0.01, model="fake", model_version="fake-1",
            context_seconds=prompt.context_seconds, prompt_chars=1,
        )

    def chapters(self, transcript, title):
        if self.fail:
            raise self.fail
        return Completion(f"### [00:00] {title}\n- x", Usage(1, 1, 0, 2), 0.01, "fake", "fake-1")


@pytest.fixture
def fake_gen():
    return FakeGenerator()


@pytest.fixture
def client(settings, monkeypatch, fake_gen):
    monkeypatch.setattr(
        service_mod, "fetch_video_details",
        lambda vid: {"video_id": vid, "title": f"Title {vid}", "duration": 120, "author": "A", "thumbnail": "t", "url": "u"},
    )
    monkeypatch.setattr(
        service_mod, "fetch_caption_track",
        lambda vid: Transcript(vid, _lines(vid), SOURCE_MANUAL, "en"),
    )
    svc = Lectura(settings, embedder=HashEmbedder(), generator_factory=lambda key: fake_gen)
    app = create_app(settings, svc)
    with TestClient(app) as c:
        yield c


def _ingest(client, url=f"https://www.youtube.com/watch?v={VID}"):
    r = client.post("/api/ingest", json={"url": url})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "collection_name" in body["config"]
    assert "gemini_api_key" not in body["config"]


def test_ingest_video_end_to_end(client):
    body = _ingest(client)
    assert body["kind"] == "video" and body["video"]["video_id"] == VID
    videos = client.get("/api/videos").json()
    assert len(videos) == 1
    v = videos[0]
    assert v["status"] == "indexed" and v["n_chunks"] == 4 and v["transcript_source"] == SOURCE_MANUAL


def test_watch_url_with_list_param_is_single_video(client):
    body = _ingest(client, f"https://www.youtube.com/watch?v={VID}&list={PL}")
    assert body["kind"] == "video"
    assert client.get("/api/playlists").json() == []


def test_ingest_playlist(client, monkeypatch):
    monkeypatch.setattr(
        service_mod, "fetch_playlist_details",
        lambda pid: {
            "playlist_id": pid, "title": "My PL",
            "videos": [
                {"video_id": v, "title": f"T {v}", "duration": 1, "author": "A", "thumbnail": "t", "url": "u"}
                for v in (VID, VID2)
            ],
        },
    )
    body = _ingest(client, f"https://www.youtube.com/playlist?list={PL}")
    assert body["kind"] == "playlist" and body["playlist"]["video_ids"] == [VID, VID2]
    pls = client.get("/api/playlists").json()
    assert len(pls) == 1 and [v["status"] for v in pls[0]["videos"]] == ["indexed", "indexed"]

    r = client.delete(f"/api/playlists/{PL}")
    assert r.status_code == 200
    assert client.get("/api/videos").json() == [] and client.get("/api/playlists").json() == []
    assert client.delete(f"/api/playlists/{PL}").status_code == 404


def test_ingest_rejects_garbage(client):
    assert client.post("/api/ingest", json={"url": "https://example.com/x"}).status_code == 400
    r = client.post("/api/ingest", json={"url": "https://youtu.be/" + VID, "whisper_model": "huge"})
    assert r.status_code == 422


def test_whisper_fallback_path(client, monkeypatch, settings):
    def no_captions(vid):
        raise CaptionsUnavailable("disabled")

    monkeypatch.setattr(service_mod, "fetch_caption_track", no_captions)

    def fake_download(vid, audio_dir):
        p = audio_dir / f"{vid}.mp3"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00")
        return p

    monkeypatch.setattr(service_mod, "download_audio", fake_download)
    monkeypatch.setattr(WhisperTranscriber, "transcribe", lambda self, path, name, **kw: _lines("w"))
    _ingest(client)
    v = client.get("/api/videos").json()[0]
    assert v["status"] == "indexed" and v["transcript_source"] == "whisper:tiny.en"
    assert list(settings.audio_dir.glob("*")) == []  # audio cleaned up


def test_failed_ingest_is_reported(client, monkeypatch):
    def no_captions(vid):
        raise CaptionsUnavailable("disabled")

    def no_audio(vid, audio_dir):
        raise RuntimeError("yt-dlp exploded")

    monkeypatch.setattr(service_mod, "fetch_caption_track", no_captions)
    monkeypatch.setattr(service_mod, "download_audio", no_audio)
    _ingest(client)
    v = client.get("/api/videos").json()[0]
    assert v["status"] == "failed" and "yt-dlp exploded" in v["error"]
    # Retry re-registers and succeeds once captions work again.
    monkeypatch.setattr(service_mod, "fetch_caption_track", lambda vid: Transcript(vid, _lines(), SOURCE_MANUAL, "en"))
    _ingest(client)
    assert client.get("/api/videos").json()[0]["status"] == "indexed"


def test_query_requires_key_and_targets(client):
    _ingest(client)
    assert client.post("/api/query", json={"video_id": VID, "query": "q"}).status_code == 401
    assert client.post("/api/query", json={"query": "q"}, headers=KEY).status_code == 400
    assert client.post("/api/query", json={"video_id": VID2, "query": "q"}, headers=KEY).status_code == 404
    assert client.post("/api/query", json={"video_id": "bad", "query": "q"}, headers=KEY).status_code == 422
    assert client.post("/api/query", json={"playlist_id": "nope", "query": "q"}, headers=KEY).status_code == 404


def test_query_returns_answer_with_sources(client, fake_gen):
    _ingest(client)
    r = client.post("/api/query", json={"video_id": VID, "query": "topic 7"}, headers=KEY)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("Answer citing [")
    assert body["sources"] and body["sources"][0]["video_id"] == VID
    assert body["sources"][0]["video_title"] == f"Title {VID}"
    assert body["retrieval"]["covered"] is True and body["usage"]["total_tokens"] == 2
    assert len(fake_gen.prompts) == 1 and len(fake_gen.prompts[0].sources) == 4


def test_query_top_k_override(client, fake_gen):
    _ingest(client)
    r = client.post("/api/query", json={"video_id": VID, "query": "topic", "top_k": 2}, headers=KEY)
    assert r.status_code == 200 and len(r.json()["sources"]) == 2


def test_query_abstains_without_llm_when_nothing_indexed(client, fake_gen, settings):
    # Register but make indexing produce no chunks by returning an empty transcript.
    import lectura.service as sm

    sm.fetch_caption_track = lambda vid: Transcript(vid, [], SOURCE_MANUAL, "en")
    _ingest(client)
    r = client.post("/api/query", json={"video_id": VID, "query": "q"}, headers=KEY)
    assert r.status_code == 200 and r.json()["abstained"] is True
    assert fake_gen.prompts == []


def test_generation_errors_map_to_status(client, fake_gen):
    _ingest(client)
    fake_gen.fail = GenerationError("bad key", status=401)
    assert client.post("/api/query", json={"video_id": VID, "query": "q"}, headers=KEY).status_code == 401
    fake_gen.fail = GenerationError("quota", status=429)
    assert client.post("/api/chapters", json={"video_id": VID}, headers=KEY).status_code == 429


def test_chapters(client):
    assert client.post("/api/chapters", json={"video_id": VID}, headers=KEY).status_code == 404
    _ingest(client)
    r = client.post("/api/chapters", json={"video_id": VID}, headers=KEY)
    assert r.status_code == 200 and r.json()["chapters"].startswith(f"### [00:00] Title {VID}")


def test_delete_video(client):
    _ingest(client)
    assert client.delete(f"/api/videos/{VID}").status_code == 200
    assert client.delete(f"/api/videos/{VID}").status_code == 404
    assert client.delete("/api/videos/short").status_code == 400
    assert client.get("/api/videos").json() == []


def test_startup_cleanup_marks_interrupted_as_failed(settings, monkeypatch):
    svc = Lectura(settings, embedder=HashEmbedder(), generator_factory=lambda k: FakeGenerator())
    svc.catalog.register_video(VID, title="t", duration=1, author="a", thumbnail="t")
    with TestClient(create_app(settings, svc)) as c:
        v = c.get("/api/videos").json()[0]
        assert v["status"] == "failed" and "restart" in v["error"]


def test_changing_the_index_config_reindexes_from_cached_transcripts(settings, monkeypatch, fake_gen):
    """Flipping the embedder or chunk size derives a new collection. The library
    must not be left claiming "indexed" against an empty one."""
    import dataclasses

    monkeypatch.setattr(
        service_mod, "fetch_video_details",
        lambda vid: {"video_id": vid, "title": "T", "duration": 1, "author": "A", "thumbnail": "t", "url": "u"},
    )
    monkeypatch.setattr(service_mod, "fetch_caption_track", lambda vid: Transcript(vid, _lines(vid), SOURCE_MANUAL, "en"))

    svc = Lectura(settings, embedder=HashEmbedder(), generator_factory=lambda k: fake_gen)
    with TestClient(create_app(settings, svc)) as c:
        _ingest(c)
        assert c.get("/api/videos").json()[0]["n_chunks"] == 4
    old_collection = svc.index.name

    # Same data dir, different chunk size -> different collection, initially empty.
    changed = dataclasses.replace(settings, index=dataclasses.replace(settings.index, child_secs=15))
    svc2 = Lectura(changed, embedder=HashEmbedder(), generator_factory=lambda k: fake_gen)
    assert svc2.index.name != old_collection
    assert len(svc2.index) == 0

    with TestClient(create_app(changed, svc2)) as c:
        v = c.get("/api/videos").json()[0]
        assert v["status"] == "indexed"
        assert v["collection"] == svc2.index.name
        assert len(svc2.index) > 0
        r = c.post("/api/query", json={"video_id": VID, "query": "topic 7"}, headers=KEY)
        assert r.status_code == 200 and r.json()["sources"]
