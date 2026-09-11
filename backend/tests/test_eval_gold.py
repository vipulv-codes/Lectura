from __future__ import annotations

import pytest

from eval.gold import (
    GoldItem,
    RankedChunk,
    Span,
    first_hit_rank,
    first_video_rank,
    is_hit,
    load_jsonl,
    save_jsonl,
    straddles_boundary,
)

V = "aaaaaaaaaaa"


def _gold(spans, vid=V, set_="human"):
    return GoldItem(qid="q1", set=set_, question="q", video_id=vid, spans=tuple(Span(*s) for s in spans))


def test_anchor_defaults_to_first_span_start():
    g = _gold([(120.0, 140.0), (60.0, 70.0)])
    assert g.anchor == 60.0 and g.answerable
    assert not _gold([], set_="unanswerable").answerable


def test_span_validation_and_set_validation():
    with pytest.raises(ValueError):
        Span(10, 5)
    with pytest.raises(ValueError):
        GoldItem(qid="x", set="nope", question="q", video_id=V)


@pytest.mark.parametrize(
    "chunk,expected",
    [
        ((100.0, 130.0), True),    # fully contains the 20 s span
        ((115.0, 125.0), True),    # inside the span, 10 s overlap
        ((135.0, 165.0), True),    # overlaps 5 s exactly
        ((136.0, 166.0), False),   # 4 s graze
        ((0.0, 120.0), False),     # touches boundary, zero overlap
        ((141.0, 170.0), False),   # after the span
    ],
)
def test_is_hit_overlap_rule(chunk, expected):
    g = _gold([(120.0, 140.0)])
    assert is_hit(V, chunk[0], chunk[1], g) is expected


def test_is_hit_short_span_uses_half_length():
    g = _gold([(120.0, 126.0)])  # 6 s span -> need 3 s
    assert is_hit(V, 123.0, 150.0, g)      # 3 s overlap
    assert not is_hit(V, 124.0, 150.0, g)  # 2 s


def test_is_hit_wrong_video_and_tolerance():
    g = _gold([(120.0, 140.0)])
    assert not is_hit("bbbbbbbbbbb", 100.0, 130.0, g)
    assert not is_hit(V, 136.0, 170.0, g)            # 4 s overlap with [120,140]
    assert is_hit(V, 136.0, 170.0, g, tol=3.0)       # widened to [117,143] -> 7 s overlap
    assert not is_hit(V, 141.0, 170.0, g, tol=3.0)   # 2 s overlap, still short of 5 s


def test_first_hit_and_video_ranks():
    g = _gold([(120.0, 140.0)])
    ranked = [
        RankedChunk("bbbbbbbbbbb", 120.0, 140.0, 0.9),
        RankedChunk(V, 0.0, 30.0, 0.8),
        RankedChunk(V, 118.0, 150.0, 0.7),
    ]
    assert first_hit_rank(ranked, g) == 2
    assert first_video_rank(ranked, g) == 1   # second distinct video
    assert first_hit_rank(ranked[:2], g) is None
    assert first_video_rank([ranked[0]], g) is None


def test_straddles_boundary():
    assert straddles_boundary(_gold([(25.0, 35.0)]), 30)
    assert not straddles_boundary(_gold([(31.0, 59.0)]), 30)
    assert not straddles_boundary(_gold([(30.0, 60.0)]), 30)  # ends exactly on a boundary
    assert straddles_boundary(_gold([(25.0, 35.0)]), 15)


def test_jsonl_roundtrip(tmp_path):
    items = [
        _gold([(1.0, 2.0)]),
        GoldItem(qid="u1", set="unanswerable", question="q", video_id=V, origin_qid="q1", tags={"kind": "swap"}),
    ]
    p = tmp_path / "g.jsonl"
    save_jsonl(p, items)
    back = load_jsonl(p)
    assert back == items
    assert back[1].origin_qid == "q1" and back[1].tags == {"kind": "swap"} and back[1].anchor is None
