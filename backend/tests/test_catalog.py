from __future__ import annotations

import json
import threading

import pytest

from lectura.catalog import (
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_PROCESSING,
    STATUS_TRANSCRIBING,
    Catalog,
    atomic_write_json,
)


def _reg(catalog: Catalog, vid="aaaaaaaaaaa", title="T"):
    return catalog.register_video(vid, title=title, duration=100, author="A", thumbnail="th")


def test_register_and_read(catalog):
    entry = _reg(catalog)
    assert entry["status"] == STATUS_PROCESSING
    assert entry["url"].endswith("v=aaaaaaaaaaa")
    assert catalog.get_video("aaaaaaaaaaa") == entry
    assert catalog.has_video("aaaaaaaaaaa")
    assert not catalog.has_video("zzzzzzzzzzz")


def test_status_transitions_and_extra_fields(catalog):
    _reg(catalog)
    assert catalog.set_status("aaaaaaaaaaa", STATUS_TRANSCRIBING)
    assert catalog.set_status("aaaaaaaaaaa", STATUS_FAILED, error="boom")
    assert catalog.get_video("aaaaaaaaaaa")["error"] == "boom"
    assert catalog.set_status(
        "aaaaaaaaaaa", STATUS_INDEXED, transcript_source="captions_manual", n_chunks=12
    )
    v = catalog.get_video("aaaaaaaaaaa")
    assert v["status"] == STATUS_INDEXED
    assert "error" not in v
    assert v["transcript_source"] == "captions_manual" and v["n_chunks"] == 12
    assert not catalog.set_status("zzzzzzzzzzz", STATUS_INDEXED)


def test_reregister_resets_failure(catalog):
    _reg(catalog)
    catalog.set_status("aaaaaaaaaaa", STATUS_FAILED, error="boom")
    entry = _reg(catalog, title="T2")
    assert entry["status"] == STATUS_PROCESSING and "error" not in entry


def test_bad_status_rejected(catalog):
    _reg(catalog)
    with pytest.raises(ValueError):
        catalog.set_status("aaaaaaaaaaa", "ready")


def test_delete_video(catalog):
    _reg(catalog)
    assert catalog.delete_video("aaaaaaaaaaa")
    assert not catalog.delete_video("aaaaaaaaaaa")
    assert catalog.videos() == {}


def test_playlists_and_dangling_removal(catalog):
    catalog.register_playlist("PL1", "One", ["a", "b", "c"])
    catalog.register_playlist("PL2", "Two", ["c", "d"])
    catalog.remove_videos_from_playlists("c")
    assert catalog.get_playlist("PL1")["video_ids"] == ["a", "b"]
    assert catalog.get_playlist("PL2")["video_ids"] == ["d"]
    catalog.remove_videos_from_playlists(["a", "d"])
    assert catalog.get_playlist("PL1")["video_ids"] == ["b"]
    assert catalog.get_playlist("PL2")["video_ids"] == []
    assert catalog.delete_playlist("PL2")
    assert not catalog.delete_playlist("PL2")
    assert list(catalog.playlists()) == ["PL1"]


def test_corrupt_file_reads_as_empty(catalog):
    catalog.videos_file.parent.mkdir(parents=True)
    catalog.videos_file.write_text("{not json", encoding="utf-8")
    assert catalog.videos() == {}


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.json"
    atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_concurrent_registration_is_lossless(catalog):
    ids = [f"{i:011d}" for i in range(40)]

    def worker(vid):
        _reg(catalog, vid)
        catalog.set_status(vid, STATUS_INDEXED)

    threads = [threading.Thread(target=worker, args=(v,)) for v in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    videos = catalog.videos()
    assert set(videos) == set(ids)
    assert all(v["status"] == STATUS_INDEXED for v in videos.values())
