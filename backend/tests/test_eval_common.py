from __future__ import annotations

import time

import pytest

from eval.common import CachedGemini, price_for
from lectura.config import GenerationConfig
from lectura.generation import Completion, GeminiGenerator, GenerationError, Usage


class _FakeGen(GeminiGenerator):
    def __init__(self, fail_first: int = 0):
        super().__init__("key", GenerationConfig(model="fake", temperature=0.0))
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, system: str, user: str) -> Completion:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise GenerationError("429 quota", status=429)
        return Completion(f"answer to {user}", Usage(3, 2, 0, 5), 0.01, "fake", "fake-v1")


def test_cache_hit_and_miss(tmp_path):
    gen = _FakeGen()
    c = CachedGemini(gen, cache_dir=tmp_path)
    a = c.complete("sys", "q1")
    b = c.complete("sys", "q1")
    assert a.text == b.text == "answer to q1" and gen.calls == 1
    assert b.usage == Usage(3, 2, 0, 5) and b.model_version == "fake-v1"
    c.complete("sys", "q2")
    assert gen.calls == 2 and c.stats == {"hits": 1, "misses": 2, "retries": 0}
    # A different generation config is a different cache key.
    other = CachedGemini(_FakeGen(), cache_dir=tmp_path)
    other.generator.cfg = GenerationConfig(model="fake", temperature=0.9)
    other.complete("sys", "q1")
    assert other.stats["misses"] == 1


def test_retry_on_429_then_success(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    gen = _FakeGen(fail_first=2)
    c = CachedGemini(gen, cache_dir=tmp_path, max_retries=3)
    out = c.complete("sys", "q")
    assert out.text == "answer to q" and gen.calls == 3 and c.stats["retries"] == 2


def test_gives_up_after_max_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    c = CachedGemini(_FakeGen(fail_first=10), cache_dir=tmp_path, max_retries=2)
    with pytest.raises(GenerationError):
        c.complete("sys", "q")


def test_rpm_throttle_spaces_calls(tmp_path):
    c = CachedGemini(_FakeGen(), cache_dir=tmp_path, rpm=6000)  # 10 ms spacing
    t0 = time.perf_counter()
    c.complete("s", "a")
    c.complete("s", "b")
    c.complete("s", "c")
    assert time.perf_counter() - t0 >= 0.02


def test_price_lookup():
    pricing = {"models": {"gemini-2.5-flash": {"input_per_m": 0.3, "output_per_m": 2.5}}}
    assert price_for("gemini-2.5-flash", pricing) == {"input_per_m": 0.3, "output_per_m": 2.5}
    assert price_for("unknown", pricing) == {"input_per_m": 0.0, "output_per_m": 0.0}
