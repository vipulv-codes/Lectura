"""Metric definitions. Pure functions over per-query records so every number
in the report is unit-tested against hand-built inputs before any model runs.

Conventions: ranks are 0-based; ``None`` means "not retrieved in the candidate
pool". All rates are fractions in [0, 1].
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

# Rank-based -------------------------------------------------------------------

def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    if not ranks:
        return float("nan")
    return sum(1 for r in ranks if r is not None and r < k) / len(ranks)


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    if not ranks:
        return float("nan")
    return sum(1.0 / (r + 1) for r in ranks if r is not None and r < k) / len(ranks)


def budget_k(budget_secs: float, child_secs: int) -> int:
    """How many child windows fit in a context budget (at least one)."""
    return max(1, int(budget_secs // child_secs))


def hit_at_budget(ranks: Sequence[int | None], budget_secs: float, child_secs: int) -> float:
    """Hit rate when the reader is allowed ``budget_secs`` of child windows:
    the fair way to compare chunk sizes (k=4 at 30 s equals k=8 at 15 s)."""
    return hit_at_k(ranks, budget_k(budget_secs, child_secs))


# Distributions ----------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, math.ceil(p / 100.0 * len(xs)) - 1))
    return float(xs[idx])


def summarize_errors(errors: Sequence[float], within: float = 10.0) -> dict[str, float]:
    if not errors:
        return {"n": 0, "median": float("nan"), "p90": float("nan"), f"frac_within_{int(within)}s": float("nan")}
    return {
        "n": len(errors),
        "median": percentile(errors, 50),
        "p90": percentile(errors, 90),
        f"frac_within_{int(within)}s": sum(1 for e in errors if e <= within) / len(errors),
    }


# Gate calibration -------------------------------------------------------------

def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Probability a random positive scores above a random negative (ties 0.5)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def threshold_at_tpr(pos: Sequence[float], target_tpr: float = 0.95) -> float:
    """Largest threshold that keeps at least ``target_tpr`` of positives."""
    if not pos:
        return float("nan")
    xs = sorted(pos, reverse=True)
    keep = max(1, math.ceil(target_tpr * len(xs)))
    return float(xs[keep - 1])


def rate_at_or_above(scores: Sequence[float], threshold: float) -> float:
    if not scores:
        return float("nan")
    return sum(1 for s in scores if s >= threshold) / len(scores)


# Statistics -------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}


def bootstrap_mean(values: Sequence[float], n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05) -> BootstrapResult:
    if not values:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    return BootstrapResult(mean(values), percentile(means, 100 * alpha / 2), percentile(means, 100 * (1 - alpha / 2)), n)


@dataclass(frozen=True)
class PairedResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int
    n: int

    @property
    def significant(self) -> bool:
        """CI excludes zero."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_diff": self.mean_diff, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "wins": self.wins, "losses": self.losses, "ties": self.ties, "n": self.n,
            "significant": self.significant,
        }


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 2000, seed: int = 0, alpha: float = 0.05) -> PairedResult:
    """CI on mean(b - a) over the same queries; positive means b is better."""
    if len(a) != len(b):
        raise ValueError("paired samples must align")
    if not a:
        return PairedResult(float("nan"), float("nan"), float("nan"), 0, 0, 0, 0)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    boot = bootstrap_mean(diffs, n_resamples, seed, alpha)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return PairedResult(boot.mean, boot.ci_low, boot.ci_high, wins, losses, len(diffs) - wins - losses, len(diffs))


def minimum_detectable_effect(n: int, p: float = 0.5, z: float = 1.96) -> float:
    """Rough MDE for a difference in proportions at n paired queries."""
    if n <= 0:
        return float("nan")
    return z * math.sqrt(2 * p * (1 - p) / n)


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pe = sum((sum(1 for x in a if x == lab) / n) * (sum(1 for y in b if y == lab) / n) for lab in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


# Citations ------------------------------------------------------------------------

def citation_validity(cited_labels: Iterable[str], valid_labels: set[str]) -> float | None:
    cited = list(cited_labels)
    if not cited:
        return None
    return sum(1 for c in cited if c in valid_labels) / len(cited)


def citation_localized(cited_seconds: Sequence[float], spans: Sequence[tuple[float, float]], tol: float = 10.0) -> bool | None:
    """Does the *first* citation land inside a gold span (± tol)?"""
    if not cited_seconds or not spans:
        return None
    first = cited_seconds[0]
    return any(s - tol <= first <= e + tol for s, e in spans)


# Cost --------------------------------------------------------------------------------

def estimate_cost_usd(usage: dict[str, int], pricing: dict[str, float]) -> float:
    """``pricing`` has per-million-token prices: input, output (thinking is
    billed as output on Gemini 2.5)."""
    inp = usage.get("prompt_tokens", 0) / 1e6 * pricing.get("input_per_m", 0.0)
    out = (usage.get("output_tokens", 0) + usage.get("thinking_tokens", 0)) / 1e6 * pricing.get("output_per_m", 0.0)
    return inp + out


# Aggregation ---------------------------------------------------------------------------

def summarize_retrieval(records: Sequence[dict[str, Any]], child_secs: int) -> dict[str, Any]:
    """Aggregate per-query retrieval records (see run_retrieval.py for the
    record schema) into the headline numbers."""
    answerable = [r for r in records if r.get("answerable", True)]
    ranks = [r.get("first_hit_rank") for r in answerable]
    vranks = [r.get("first_video_rank") for r in answerable]
    errors = [r["ts_error"] for r in answerable if r.get("ts_error") is not None]
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    out: dict[str, Any] = {"n": len(answerable), "n_total": len(records), "child_secs": child_secs}
    for k in (1, 3, 5, 10):
        out[f"hit@{k}"] = hit_at_k(ranks, k)
    out["mrr@10"] = mrr_at_k(ranks, 10)
    for budget in (120, 240):
        out[f"hit@{budget}s"] = hit_at_budget(ranks, budget, child_secs)
    out["not_in_pool"] = (sum(1 for r in ranks if r is None) / len(ranks)) if ranks else float("nan")
    for k in (1, 3, 5):
        out[f"video_hit@{k}"] = hit_at_k(vranks, k)
    out["ts_error"] = summarize_errors(errors)
    out["latency_s"] = {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)} if latencies else {}
    return out


def slice_records(records: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        value = r.get(key)
        if value is None and "tags" in r:
            value = r["tags"].get(key)
        groups.setdefault(str(value), []).append(r)
    return groups
