from __future__ import annotations

import pytest

from lectura.youtube import extract_playlist_id, extract_video_id, parse_youtube_url

VID = "dQw4w9WgXcQ"
PL = "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/live/{VID}",
        f"www.youtube.com/watch?v={VID}",
        f"  {VID}  ",
    ],
)
def test_extract_video_id(url):
    assert extract_video_id(url) == VID


def test_watch_inside_playlist_is_a_video_request():
    p = parse_youtube_url(f"https://www.youtube.com/watch?v={VID}&list={PL}&index=3")
    assert p.video_id == VID and p.playlist_id == PL
    assert p.kind == "video"


def test_pure_playlist_url():
    p = parse_youtube_url(f"https://www.youtube.com/playlist?list={PL}")
    assert p.video_id is None and p.playlist_id == PL
    assert p.kind == "playlist"
    assert extract_playlist_id(f"https://www.youtube.com/playlist?list={PL}") == PL


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/dQw4w9WgXcQ",          # non-YouTube host with an 11-char segment
        "https://evil.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=../../etc",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/",
        "not a url",
        "",
    ],
)
def test_rejects_invalid(url):
    with pytest.raises(ValueError):
        extract_video_id(url)
    assert parse_youtube_url(url).kind is None or parse_youtube_url(url).video_id is None


def test_rejects_bad_playlist_id():
    with pytest.raises(ValueError):
        extract_playlist_id("https://www.youtube.com/playlist?list=a")
    with pytest.raises(ValueError):
        extract_playlist_id("https://www.youtube.com/playlist?list=../x")
