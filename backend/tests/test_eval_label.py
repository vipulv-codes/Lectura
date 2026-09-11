from __future__ import annotations

import pytest

from eval.label import parse_spans, position_of, snap
from lectura.chunking import Line

# Mirrors the real caption timings around h_001 in aircAruvnKk.
LINES = [
    Line("What are the neurons, and in what sense are they linked together?", 168.52, 3.14),
    Line("Right now when I say neuron all I want you to think about is a thing that holds a number,", 172.50, 5.52),
    Line("specifically a number between 0 and 1.", 178.02, 2.42),
    Line("It's really not more than that.", 180.68, 1.88),
    Line("For example the network starts with a bunch of neurons corresponding to", 183.78, 5.04),
]


class TestSnap:
    def test_exact_line_start(self):
        assert snap(LINES, 172.50).text.startswith("Right now")

    def test_time_inside_a_line(self):
        assert snap(LINES, 175.0).text.startswith("Right now")
        assert snap(LINES, 179.9).text.startswith("specifically")

    def test_time_in_a_gap_picks_nearest_start(self):
        # 182.9 sits between lines; nearest start is 183.78, not 180.68.
        assert snap(LINES, 182.9).text.startswith("For example")

    def test_typed_label_wins_over_the_containing_line(self):
        # Typing 02:58 (=178.0) must give the line LABELLED [02:58], which
        # starts at 178.02 - even though 178.0 still lies inside line [02:52].
        assert snap(LINES, 178.0).text.startswith("specifically")
        assert snap(LINES, 178.5).text.startswith("specifically")


class TestParseSpans:
    def test_range_runs_first_start_to_last_end(self):
        (s,) = parse_spans("02:52 02:58", LINES)
        assert (s.start, s.end) == (172.5, 180.44)

    def test_single_timestamp_is_one_line(self):
        (s,) = parse_spans("02:58", LINES)
        assert (s.start, s.end) == (178.02, 180.44)

    @pytest.mark.parametrize("text", ["02:52 02:58", "[02:52] [02:58]", "02:52-02:58", "02:52 to 02:58", " 02:52   02:58 "])
    def test_accepted_formats(self, text):
        (s,) = parse_spans(text, LINES)
        assert (s.start, s.end) == (172.5, 180.44)

    def test_reversed_order_is_normalised(self):
        (s,) = parse_spans("02:58 02:52", LINES)
        assert (s.start, s.end) == (172.5, 180.44)

    def test_multiple_spans(self):
        a, b = parse_spans("02:48 02:52, 03:03", LINES)
        assert (a.start, a.end) == (168.52, 178.02)
        assert (b.start, b.end) == (183.78, 188.82)

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_spans("banana", LINES)
        assert parse_spans("  ,  ", LINES) == []


def test_position_of():
    assert position_of(10, 900) == "early"
    assert position_of(400, 900) == "mid"
    assert position_of(800, 900) == "late"
