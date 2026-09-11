from __future__ import annotations

import math

import pytest

from eval.metrics import (
    agreement,
    auroc,
    bootstrap_mean,
    budget_k,
    citation_localized,
    citation_validity,
    cohen_kappa,
    estimate_cost_usd,
    hit_at_budget,
    hit_at_k,
    minimum_detectable_effect,
    mrr_at_k,
    paired_bootstrap,
    percentile,
    rate_at_or_above,
    slice_records,
    summarize_errors,
    summarize_retrieval,
    threshold_at_tpr,
)

RANKS = [0, 2, None, 9, 4]


def test_hit_and_mrr():
    assert hit_at_k(RANKS, 1) == 0.2
    assert hit_at_k(RANKS, 3) == 0.4
    assert hit_at_k(RANKS, 10) == 0.8
    assert mrr_at_k(RANKS, 10) == pytest.approx((1 + 1 / 3 + 1 / 10 + 1 / 5) / 5)
    assert mrr_at_k(RANKS, 3) == pytest.approx((1 + 1 / 3) / 5)
    assert math.isnan(hit_at_k([], 1))


def test_budget_hits_compare_chunk_sizes_fairly():
    assert budget_k(120, 30) == 4 and budget_k(120, 15) == 8 and budget_k(120, 60) == 2
    assert budget_k(10, 30) == 1
    ranks = [3, 5, 1]
    assert hit_at_budget(ranks, 120, 30) == pytest.approx(2 / 3)   # k=4 -> ranks 3,1
    assert hit_at_budget(ranks, 120, 15) == 1.0                    # k=8
    assert hit_at_budget(ranks, 120, 60) == pytest.approx(1 / 3)   # k=2


def test_percentile_and_error_summary():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == 5 and percentile(xs, 90) == 9 and percentile(xs, 100) == 10
    s = summarize_errors([0, 3, 12, 40], within=10)
    assert s["n"] == 4 and s["median"] == 3 and s["frac_within_10s"] == 0.5
    assert math.isnan(summarize_errors([])["median"])


def test_auroc_and_threshold():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    pos = [0.9, 0.85, 0.8, 0.7, 0.3]
    thr = threshold_at_tpr(pos, 0.8)          # keep 4 of 5
    assert thr == 0.7
    assert rate_at_or_above(pos, thr) == 0.8
    assert rate_at_or_above([0.2, 0.75], thr) == 0.5


def test_bootstrap_is_deterministic_and_sane():
    vals = [1.0] * 10 + [0.0] * 10
    b1 = bootstrap_mean(vals, 500, seed=1)
    b2 = bootstrap_mean(vals, 500, seed=1)
    assert b1 == b2 and b1.mean == 0.5 and b1.ci_low < 0.5 < b1.ci_high and b1.n == 20


def test_paired_bootstrap_detects_clear_win_and_ties():
    a = [0.0] * 30
    b = [1.0] * 30
    r = paired_bootstrap(a, b, 500, seed=0)
    assert r.mean_diff == 1.0 and r.wins == 30 and r.losses == 0 and r.significant
    same = paired_bootstrap(a, a, 200)
    assert same.ties == 30 and not same.significant
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 2])


def test_mde_shrinks_with_n():
    assert minimum_detectable_effect(50) > minimum_detectable_effect(300)
    # 1.96 * sqrt(2 * 0.25 / 50) ~= 0.196: about 20 pp at n=50, ~8 pp at n=300.
    assert 0.19 < minimum_detectable_effect(50) < 0.20
    assert 0.07 < minimum_detectable_effect(300) < 0.09


def test_kappa_and_agreement():
    a = ["y", "y", "n", "n"]
    assert cohen_kappa(a, a) == 1.0 and agreement(a, a) == 1.0
    assert cohen_kappa(a, ["n", "n", "y", "y"]) == -1.0
    assert cohen_kappa(["y", "y"], ["y", "y"]) == 1.0
    assert agreement(["y", "n"], ["y", "y"]) == 0.5


def test_citation_checks():
    assert citation_validity(["01:00", "02:00"], {"01:00"}) == 0.5
    assert citation_validity([], {"01:00"}) is None
    assert citation_localized([65.0], [(60.0, 70.0)]) is True
    assert citation_localized([85.0], [(60.0, 70.0)], tol=10) is False
    assert citation_localized([79.0, 5.0], [(60.0, 70.0)], tol=10) is True
    assert citation_localized([], [(1, 2)]) is None


def test_cost_estimate():
    usage = {"prompt_tokens": 1_000_000, "output_tokens": 500_000, "thinking_tokens": 500_000}
    assert estimate_cost_usd(usage, {"input_per_m": 0.3, "output_per_m": 2.5}) == pytest.approx(0.3 + 2.5)


def test_summarize_and_slice():
    recs = [
        {"qid": "a", "answerable": True, "first_hit_rank": 0, "first_video_rank": 0, "ts_error": 2.0, "latency_s": 0.01, "tags": {"lexical": "high"}},
        {"qid": "b", "answerable": True, "first_hit_rank": None, "first_video_rank": 1, "ts_error": None, "latency_s": 0.03, "tags": {"lexical": "low"}},
        {"qid": "u", "answerable": False, "first_hit_rank": None, "first_video_rank": None, "latency_s": 0.02, "tags": {}},
    ]
    s = summarize_retrieval(recs, child_secs=30)
    assert s["n"] == 2 and s["n_total"] == 3
    assert s["hit@1"] == 0.5 and s["mrr@10"] == 0.5 and s["not_in_pool"] == 0.5
    assert s["hit@120s"] == 0.5 and s["video_hit@3"] == 1.0
    assert s["ts_error"]["n"] == 1 and s["latency_s"]["p50"] == 0.02
    groups = slice_records(recs, "lexical")
    assert set(groups) == {"high", "low", "None"} and len(groups["None"]) == 1
