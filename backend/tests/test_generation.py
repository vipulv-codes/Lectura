from __future__ import annotations

import pytest

from lectura.chunking import chunk_transcript, format_ts
from lectura.config import GenerationConfig
from lectura.generation import (
    ABSTAIN_SENTINEL,
    GeminiGenerator,
    GenerationError,
    Prompt,
    Usage,
    build_context,
    build_prompt,
    no_context_answer,
    parse_answer,
)
from lectura.index import Hit
from lectura.transcripts import SOURCE_MANUAL, Transcript, TranscriptStore

VID = "aaaaaaaaaaa"


@pytest.fixture
def store(settings, lines):
    s = TranscriptStore(settings.transcripts_dir)
    s.save(Transcript(VID, lines, SOURCE_MANUAL, "en"))
    return s


def _hits(lines, *indices):
    chunks = {c.index: c for c in chunk_transcript(lines, 30, VID)}
    return [Hit(chunks[i], 0.9 - r * 0.1, r) for r, i in enumerate(indices)]


def test_context_merges_overlapping_parents(store, lines):
    hits = _hits(lines, 0, 1)  # windows centred at 15 and 45 with 60s parents -> [0,60) and [15,75)
    blocks = build_context(hits, store, {VID: "Lec"}, parent_secs=60)
    assert len(blocks) == 1
    b = blocks[0]
    assert (b.start, b.end) == (0.0, 75.0)
    assert b.matched == (0.0, 30.0)
    assert [ln.start for ln in b.lines] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    assert "[00:00] line 0" in b.render() and "Lec" in b.render()


def test_context_keeps_distant_windows_separate_and_orders(store, lines):
    hits = _hits(lines, 3, 0)  # rank 0 is chunk 3 (late), rank 1 is chunk 0 (early)
    time_order = build_context(hits, store, {VID: "Lec"}, parent_secs=20, order="time")
    assert [b.start for b in time_order] == [5.0, 95.0]
    rank_order = build_context(hits, store, {VID: "Lec"}, parent_secs=20, order="rank")
    assert [b.start for b in rank_order] == [95.0, 5.0]


def test_context_without_transcript_falls_back_to_chunk_text(settings, lines):
    empty_store = TranscriptStore(settings.transcripts_dir / "nope")
    blocks = build_context(_hits(lines, 2), empty_store, {}, 60)
    assert len(blocks) == 1 and blocks[0].lines[0].text == "line 6 line 7 line 8"


def test_build_prompt_valid_timestamps_and_sources(store, lines):
    hits = _hits(lines, 2)
    p = build_prompt("what is line 7?", hits, store, {VID: "Lec"}, parent_secs=30)
    assert isinstance(p, Prompt)
    assert p.valid_timestamps == {format_ts(s) for s in (60.0, 70.0, 80.0)}
    assert p.context_seconds == 30.0
    assert ABSTAIN_SENTINEL in p.system and "what is line 7?" in p.user
    assert len(p.sources) == 1
    src = p.sources[0]
    assert src.timestamp == "01:00" and src.seconds == 60.0 and src.chunk_id == f"{VID}:2"


def test_parse_answer_flags_invalid_citations_and_abstention(store, lines):
    p = build_prompt("q", _hits(lines, 2), store, {}, 30)
    abst, cites, invalid = parse_answer("See [01:10] and [59:59].", p)
    assert abst is False
    assert cites == (("01:10", 70.0), ("59:59", 3599.0))
    assert invalid == ("59:59",)
    abst, cites, invalid = parse_answer(f"{ABSTAIN_SENTINEL} The segments cover X.", p)
    assert abst is True and cites == () and invalid == ()


def test_no_context_answer():
    a = no_context_answer("q")
    assert a.abstained and a.text.startswith(ABSTAIN_SENTINEL) and a.sources == ()
    assert a.to_dict()["usage"] == Usage().to_dict()


def test_generator_requires_key():
    with pytest.raises(GenerationError) as ei:
        GeminiGenerator("", GenerationConfig())
    assert ei.value.status == 401


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.model_version = "fake-001"

        class _U:
            prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count = 10, 5, 2, 17

        self.usage_metadata = _U()


class _FakeModels:
    def __init__(self, text_or_exc):
        self._t = text_or_exc
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if isinstance(self._t, Exception):
            raise self._t
        return _FakeResponse(self._t)


def _fake_generator(text_or_exc, cfg=None):
    g = GeminiGenerator("key", cfg or GenerationConfig())

    class _C:
        models = _FakeModels(text_or_exc)

    g._client = _C()
    return g


def test_generator_answer_parses_usage_and_citations(store, lines):
    g = _fake_generator("Line seven is at [01:10].")
    p = build_prompt("q", _hits(lines, 2), store, {VID: "Lec"}, 30)
    a = g.answer(p)
    assert a.text.endswith("[01:10].") and a.citations == (("01:10", 70.0),)
    assert a.invalid_citations == () and not a.abstained
    assert a.usage == Usage(10, 5, 2, 17) and a.model_version == "fake-001"
    assert a.to_dict()["sources"][0]["video_title"] == "Lec"
    model, contents, config = g._client.models.calls[0]
    assert model == "gemini-2.5-flash" and contents == [p.user]
    assert config.system_instruction == p.system and config.temperature == 0.2


def test_generator_thinking_budget_is_forwarded():
    g = _fake_generator("ok", GenerationConfig(thinking_budget=0, max_output_tokens=50))
    g.complete("sys", "user")
    config = g._client.models.calls[0][2]
    assert config.thinking_config.thinking_budget == 0 and config.max_output_tokens == 50


@pytest.mark.parametrize(
    "exc,status",
    [
        (RuntimeError("API key not valid. Please pass a valid API key."), 401),
        (RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"), 429),
        (RuntimeError("something else broke"), 502),
    ],
)
def test_generator_error_classification(exc, status):
    g = _fake_generator(exc)
    with pytest.raises(GenerationError) as ei:
        g.complete("s", "u")
    assert ei.value.status == status


def test_generator_empty_response_is_error():
    g = _fake_generator("")
    with pytest.raises(GenerationError):
        g.complete("s", "u")


def test_chapters_prompt_contains_transcript(store, lines):
    g = _fake_generator("### [00:00] Intro\n- x")
    c = g.chapters(store.load(VID), "Lec")
    assert c.text.startswith("### [00:00]")
    prompt = g._client.models.calls[0][1][0]
    assert "[00:10] line 1" in prompt and "Lecture title: Lec" in prompt
