from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from lectura.config import GenerationConfig, IndexConfig, RetrievalConfig, Settings


def test_from_env_with_no_vars_equals_the_dataclass_defaults():
    """Regression guard: from_env() used to hardcode its fallbacks, so changing
    a dataclass default silently had no effect on the running app."""
    s = Settings.from_env(env={})
    assert s.index == IndexConfig()
    assert s.retrieval == RetrievalConfig()
    assert s.generation == GenerationConfig()
    assert s.whisper_model == Settings.whisper_model
    assert s.max_playlist_videos == Settings.max_playlist_videos


def test_every_index_and_retrieval_field_is_env_overridable():
    env = {
        "LECTURA_CHILD_SECS": "15",
        "LECTURA_EMBEDDING_MODEL": "x/y",
        "LECTURA_DOC_PREFIX": "passage: ",
        "LECTURA_QUERY_PREFIX": "query: ",
        "LECTURA_SPACE": "l2",
        "LECTURA_EF_SEARCH": "50",
        "LECTURA_STRATEGY": "hybrid",
        "LECTURA_CANDIDATE_K": "30",
        "LECTURA_RERANKER": "none",
        "LECTURA_TOP_K": "6",
        "LECTURA_MIN_SCORE": "0.4",
        "LECTURA_PARENT_SECS": "240",
        "LECTURA_THINKING_BUDGET": "0",
    }
    s = Settings.from_env(env=env)
    assert s.index.child_secs == 15 and s.index.embedding_model == "x/y"
    assert s.index.doc_prefix == "passage: " and s.index.query_prefix == "query: "
    assert s.index.space == "l2" and s.index.ef_search == 50
    assert s.retrieval.strategy == "hybrid" and s.retrieval.candidate_k == 30
    assert s.retrieval.reranker is None          # "none" clears it
    assert s.retrieval.top_k == 6 and s.retrieval.min_score == 0.4
    assert s.retrieval.parent_secs == 240
    assert s.generation.thinking_budget == 0


def test_defaults_are_the_measured_winners():
    """These were selected by the Stage A ablation; see eval/RESULTS.md."""
    assert IndexConfig().embedding_model == "BAAI/bge-small-en-v1.5"
    assert IndexConfig().query_prefix.startswith("Represent this sentence")
    assert RetrievalConfig().reranker == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert IndexConfig().child_secs == 30       # 60s won MRR but lost on Hit@120s
    assert RetrievalConfig().strategy == "dense"  # hybrid gain did not survive the non-lexical slice


def test_changing_the_embedder_changes_the_collection():
    a = IndexConfig().collection_name()
    b = IndexConfig(embedding_model="sentence-transformers/all-MiniLM-L6-v2", query_prefix="").collection_name()
    assert a != b


@pytest.mark.parametrize("cfg", [IndexConfig, RetrievalConfig, GenerationConfig])
def test_configs_are_frozen(cfg):
    obj = cfg()
    with pytest.raises(FrozenInstanceError):
        setattr(obj, fields(cfg)[0].name, None)


def test_prefix_env_vars_keep_their_trailing_space():
    """e5 expects "query: " / "passage: "; stripping the trailing space would
    glue the prefix onto the first word and quietly degrade every embedding."""
    s = Settings.from_env(env={"LECTURA_QUERY_PREFIX": "query: ", "LECTURA_DOC_PREFIX": "passage: "})
    assert s.index.query_prefix == "query: "
    assert s.index.doc_prefix == "passage: "
