from __future__ import annotations

import pytest

from lectura.chunking import (
    Line,
    chunk_transcript,
    find_timestamps,
    format_lines,
    format_ts,
    lines_in_range,
    normalize_lines,
    parent_window,
    parse_ts,
    transcript_duration,
)


class TestTimestamps:
    @pytest.mark.parametrize(
        "secs,label",
        [(0, "00:00"), (5.9, "00:05"), (59, "00:59"), (60, "01:00"), (754, "12:34"),
         (3599, "59:59"), (3600, "01:00:00"), (3723.4, "01:02:03"), (-3, "00:00")],
    )
    def test_format_ts(self, secs, label):
        assert format_ts(secs) == label

    @pytest.mark.parametrize(
        "label,secs",
        [("12:34", 754), ("[12:34]", 754), ("01:02:03", 3723), ("[ 01:02:03 ]", 3723), ("0:05", 5)],
    )
    def test_parse_ts(self, label, secs):
        assert parse_ts(label) == secs

    @pytest.mark.parametrize("bad", ["", "12", "12:3", "1:2:3:4", "abc", "[12:34, 15:20]"])
    def test_parse_ts_rejects(self, bad):
        with pytest.raises(ValueError):
            parse_ts(bad)

    def test_format_parse_roundtrip(self):
        for s in (0, 1, 59, 60, 3599, 3600, 7325):
            assert parse_ts(format_ts(s)) == s

    def test_find_timestamps_in_answer(self):
        text = "Gradient descent is introduced at [12:34] and revisited [1:02:03]. Not [12:34, 15:20]."
        assert find_timestamps(text) == [("12:34", 754), ("01:02:03", 3723)]

    def test_find_timestamps_none(self):
        assert find_timestamps("no citations here") == []


class TestNormalize:
    def test_sorts_strips_and_drops_empty(self):
        raw = [Line("  b \n c ", 10.0, 2.0), Line("   ", 5.0, 1.0), Line("a", 0.0, 1.0)]
        out = normalize_lines(raw)
        assert [ln.text for ln in out] == ["a", "b c"]
        assert [ln.start for ln in out] == [0.0, 10.0]

    def test_clamps_negative(self):
        out = normalize_lines([Line("x", -1.0, -2.0)])
        assert out == [Line("x", 0.0, 0.0)]

    def test_from_dict_and_object(self):
        class Snip:
            text, start, duration = "hi", 1.0, 2.0

        assert Line.from_dict({"text": "hi", "start": 1, "duration": 2}) == Line("hi", 1.0, 2.0)
        assert Line.from_dict(Snip()) == Line("hi", 1.0, 2.0)

    def test_duration(self, lines):
        assert transcript_duration(lines) == 98.0
        assert transcript_duration([]) == 0.0


class TestChunking:
    def test_empty(self):
        assert chunk_transcript([], 30, "v") == []
        assert chunk_transcript([Line("   ", 0, 1)], 30, "v") == []

    def test_each_line_assigned_exactly_once(self, lines):
        chunks = chunk_transcript(lines, 30, "v")
        texts = " ".join(c.text for c in chunks).split()
        assert sorted(texts) == sorted(" ".join(ln.text for ln in lines).split())

    def test_boundary_line_goes_to_window_containing_its_start(self):
        # A line starting exactly at 30.0 belongs to window 1, not window 0,
        # even though the previous implementation's closed-interval overlap
        # test put it in both.
        ls = [Line("a", 25.0, 5.0), Line("b", 30.0, 5.0)]
        chunks = chunk_transcript(ls, 30, "v")
        assert [(c.index, c.text) for c in chunks] == [(0, "a"), (1, "b")]

    def test_grid_and_snippet_extents(self, lines):
        chunks = chunk_transcript(lines, 30, "v")
        assert [c.index for c in chunks] == [0, 1, 2, 3]
        c1 = chunks[1]
        assert (c1.start, c1.end) == (30.0, 60.0)
        assert c1.snippet_start == 30.0
        assert c1.snippet_end == 58.0  # line at 50 lasts until 58
        assert c1.chunk_id == "v:1"
        assert c1.mid == 45.0

    def test_long_line_extent_may_exceed_grid(self):
        chunks = chunk_transcript([Line("long", 20.0, 40.0)], 30, "v")
        assert len(chunks) == 1 and chunks[0].snippet_end == 60.0 and chunks[0].end == 30.0

    def test_gaps_produce_no_empty_chunks(self):
        ls = [Line("a", 0.0, 1.0), Line("b", 1000.0, 1.0)]
        chunks = chunk_transcript(ls, 30, "v")
        assert [c.index for c in chunks] == [0, 33]

    def test_exact_multiple_no_spurious_window(self):
        ls = [Line("a", 0.0, 30.0), Line("b", 30.0, 30.0)]
        assert len(chunk_transcript(ls, 30, "v")) == 2

    @pytest.mark.parametrize("child", [15, 30, 60])
    def test_child_size_changes_boundaries_not_content(self, lines, child):
        chunks = chunk_transcript(lines, child, "v")
        assert all(c.end - c.start == child for c in chunks)
        assert " ".join(c.text for c in chunks) == " ".join(ln.text for ln in lines)

    def test_rejects_bad_child(self):
        with pytest.raises(ValueError):
            chunk_transcript([], 0, "v")


class TestWindows:
    def test_lines_in_range_half_open(self, lines):
        got = lines_in_range(lines, 10.0, 30.0)
        assert [ln.start for ln in got] == [10.0, 20.0]

    def test_parent_window_centered_and_clamped(self, lines):
        s, e, got = parent_window(lines, center=45.0, parent_secs=60)
        assert (s, e) == (15.0, 75.0)
        assert [ln.start for ln in got] == [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

        s, e, got = parent_window(lines, center=5.0, parent_secs=60)
        assert (s, e) == (0.0, 60.0)
        assert [ln.start for ln in got] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    def test_parent_window_rejects_bad_size(self, lines):
        with pytest.raises(ValueError):
            parent_window(lines, 0, 0)

    def test_format_lines(self):
        assert format_lines([Line("hi", 754, 1), Line("yo", 3723, 1)]) == "[12:34] hi\n[01:02:03] yo"
