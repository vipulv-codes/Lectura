from __future__ import annotations

from eval.failures import classify_generation, classify_retrieval


def _r(**kw):
    base = {"answerable": True, "first_hit_rank": None, "first_video_rank": 0}
    base.update(kw)
    return base


def test_retrieval_classes():
    assert classify_retrieval(_r(first_hit_rank=0), 4) == "ok"
    assert classify_retrieval(_r(first_hit_rank=3), 4) == "ok"
    assert classify_retrieval(_r(first_hit_rank=4), 4) == "ranking_miss"
    assert classify_retrieval(_r(first_hit_rank=None, first_video_rank=2), 4) == "semantic_miss"
    assert classify_retrieval(_r(first_hit_rank=None, first_video_rank=None), 4) == "wrong_video"
    assert classify_retrieval(_r(answerable=False), 4) == "n/a"


def test_generation_classes():
    ok = {"answerable": True, "abstained": False, "n_citations": 2, "invalid_citations": [], "citation_localized": True}
    assert classify_generation(ok) == ["ok"]
    assert classify_generation({**ok, "abstained": True}) == ["false_abstention"]
    assert classify_generation({**ok, "n_citations": 0}) == ["no_citation"]
    assert classify_generation({**ok, "invalid_citations": ["59:59"]}) == ["invalid_citation"]
    assert classify_generation({**ok, "citation_localized": False}) == ["mislocalized"]
    assert set(classify_generation({**ok, "invalid_citations": ["9:99"], "citation_localized": False})) == {
        "invalid_citation", "mislocalized"
    }
    assert classify_generation({"answerable": False, "abstained": False}) == ["missed_abstention"]
    assert classify_generation({"answerable": False, "abstained": True}) == ["ok"]
