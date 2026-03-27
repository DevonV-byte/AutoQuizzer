# Unit tests for Database_production/metadata_tagger.py.
# Verifies metadata enrichment logic without making any live API calls.
# Tests cover: happy-path classification, JSON parse failures, partial LLM
# responses, invalid difficulty values, and source filename normalisation.
#
# Created: 2026-03-27
# Author: Devon Vanaenrode

# --- Imports ---
import json
from unittest.mock import MagicMock
import pytest

from Database_production.metadata_tagger import (
    _build_classification_prompt,
    _parse_llm_response,
    _tag_batch,
    tag_chunks_with_metadata,
    DEFAULT_TOPIC_TAXONOMY,
    FALLBACK_METADATA,
)


# --- Helpers ---

def _make_chunk(text: str, source: str = "some/path/to/file.txt") -> MagicMock:
    """Creates a minimal mock LangChain Document chunk."""
    chunk = MagicMock()
    chunk.page_content = text
    chunk.metadata = {"source": source}
    return chunk


def _make_llm_response(items: list) -> MagicMock:
    """Wraps a list of dicts as a mock LLM response with a .content JSON string."""
    response = MagicMock()
    response.content = json.dumps(items)
    return response


# --- Tests: _build_classification_prompt ---

def test_build_prompt_contains_all_indices():
    """Each chunk index should appear as [0], [1], … in the prompt."""
    chunks = [_make_chunk(f"chunk text {i}") for i in range(3)]
    prompt = _build_classification_prompt(chunks, DEFAULT_TOPIC_TAXONOMY)
    for i in range(3):
        assert f"[{i}]" in prompt


def test_build_prompt_contains_taxonomy():
    """The prompt should include every topic from the taxonomy."""
    chunks = [_make_chunk("some text")]
    prompt = _build_classification_prompt(chunks, DEFAULT_TOPIC_TAXONOMY)
    for topic in DEFAULT_TOPIC_TAXONOMY:
        assert topic in prompt


def test_build_prompt_truncates_long_chunks():
    """Chunk text longer than 400 chars should be truncated in the prompt."""
    long_text = "x" * 1000
    chunks = [_make_chunk(long_text)]
    prompt = _build_classification_prompt(chunks, DEFAULT_TOPIC_TAXONOMY)
    # The prompt should not contain more than 400 consecutive 'x' chars
    assert "x" * 401 not in prompt


# --- Tests: _parse_llm_response ---

def test_parse_valid_response():
    """A well-formed JSON response should be parsed into the expected dict."""
    response_text = json.dumps([
        {"index": 0, "topic_cluster": "LLM Fundamentals", "difficulty_tier": "beginner"},
        {"index": 1, "topic_cluster": "RAG Architecture",  "difficulty_tier": "advanced"},
    ])
    result = _parse_llm_response(response_text, batch_size=2)
    assert result[0] == {"topic_cluster": "LLM Fundamentals", "difficulty_tier": "beginner"}
    assert result[1] == {"topic_cluster": "RAG Architecture",  "difficulty_tier": "advanced"}


def test_parse_strips_markdown_fences():
    """Markdown code fences around the JSON should be stripped before parsing."""
    inner = json.dumps([{"index": 0, "topic_cluster": "LangChain", "difficulty_tier": "intermediate"}])
    response_text = f"```json\n{inner}\n```"
    result = _parse_llm_response(response_text, batch_size=1)
    assert result[0]["topic_cluster"] == "LangChain"


def test_parse_invalid_json_returns_empty_dict():
    """Unparseable JSON should return an empty dict (not raise)."""
    result = _parse_llm_response("this is not json", batch_size=2)
    assert result == {}


def test_parse_wrong_type_returns_empty_dict():
    """A JSON object (not array) should return an empty dict."""
    result = _parse_llm_response(json.dumps({"index": 0}), batch_size=1)
    assert result == {}


def test_parse_invalid_difficulty_applies_fallback():
    """An unrecognised difficulty_tier value should be replaced with the fallback."""
    response_text = json.dumps([
        {"index": 0, "topic_cluster": "Evaluation", "difficulty_tier": "expert"}
    ])
    result = _parse_llm_response(response_text, batch_size=1)
    assert result[0]["difficulty_tier"] == FALLBACK_METADATA["difficulty_tier"]


def test_parse_partial_response():
    """A response covering fewer indices than batch_size should only populate those indices."""
    response_text = json.dumps([
        {"index": 0, "topic_cluster": "Fine-tuning", "difficulty_tier": "advanced"}
    ])
    result = _parse_llm_response(response_text, batch_size=3)
    assert 0 in result
    assert 1 not in result
    assert 2 not in result


def test_parse_out_of_range_index_skipped():
    """An index outside [0, batch_size) should be silently dropped."""
    response_text = json.dumps([
        {"index": 5, "topic_cluster": "LLM Fundamentals", "difficulty_tier": "beginner"}
    ])
    result = _parse_llm_response(response_text, batch_size=3)
    assert result == {}


# --- Tests: _tag_batch ---

def test_tag_batch_success():
    """_tag_batch should return parsed tags when the LLM responds correctly."""
    chunks = [_make_chunk("intro to LLMs"), _make_chunk("advanced RAG techniques")]
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response([
        {"index": 0, "topic_cluster": "LLM Fundamentals", "difficulty_tier": "beginner"},
        {"index": 1, "topic_cluster": "RAG Architecture",  "difficulty_tier": "advanced"},
    ])
    result = _tag_batch(llm, chunks, DEFAULT_TOPIC_TAXONOMY)
    assert result[0]["topic_cluster"] == "LLM Fundamentals"
    assert result[1]["difficulty_tier"] == "advanced"


def test_tag_batch_retries_on_exception(monkeypatch):
    """_tag_batch should retry on API exceptions and return {} after all retries fail."""
    monkeypatch.setattr("Database_production.metadata_tagger.RETRY_DELAY_SECONDS", 0)
    chunks = [_make_chunk("some text")]
    llm = MagicMock()
    llm.invoke.side_effect = Exception("network error")
    result = _tag_batch(llm, chunks, DEFAULT_TOPIC_TAXONOMY)
    assert result == {}
    assert llm.invoke.call_count == 3  # initial + MAX_RETRIES=2


def test_tag_batch_succeeds_on_second_attempt(monkeypatch):
    """_tag_batch should succeed if the LLM recovers on a retry."""
    monkeypatch.setattr("Database_production.metadata_tagger.RETRY_DELAY_SECONDS", 0)
    chunks = [_make_chunk("vector store concepts")]
    llm = MagicMock()
    good_response = _make_llm_response([
        {"index": 0, "topic_cluster": "Vector Databases", "difficulty_tier": "intermediate"}
    ])
    llm.invoke.side_effect = [Exception("timeout"), good_response]
    result = _tag_batch(llm, chunks, DEFAULT_TOPIC_TAXONOMY)
    assert result[0]["topic_cluster"] == "Vector Databases"


# --- Tests: tag_chunks_with_metadata ---

def test_tag_chunks_populates_all_metadata_fields():
    """All three metadata fields should be present on every chunk after tagging."""
    chunks = [_make_chunk("intro text"), _make_chunk("advanced text")]
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response([
        {"index": 0, "topic_cluster": "Prompt Engineering", "difficulty_tier": "beginner"},
        {"index": 1, "topic_cluster": "Agents and Tools",   "difficulty_tier": "advanced"},
    ])
    result = tag_chunks_with_metadata(chunks, llm, batch_size=10)
    assert result[0].metadata["topic_cluster"] == "Prompt Engineering"
    assert result[0].metadata["difficulty_tier"] == "beginner"
    assert result[1].metadata["topic_cluster"] == "Agents and Tools"
    assert result[1].metadata["difficulty_tier"] == "advanced"


def test_tag_chunks_normalises_source_path():
    """source metadata should be reduced to the bare filename, not the full path."""
    chunks = [_make_chunk("some content", source="IBM RAG and Agentic AI/Module 1/Labs/notes.ipynb")]
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response([
        {"index": 0, "topic_cluster": "LangChain", "difficulty_tier": "intermediate"}
    ])
    tag_chunks_with_metadata(chunks, llm)
    assert chunks[0].metadata["source"] == "notes.ipynb"


def test_tag_chunks_applies_fallback_on_llm_failure(monkeypatch):
    """Chunks in a failed batch should receive FALLBACK_METADATA values."""
    monkeypatch.setattr("Database_production.metadata_tagger.RETRY_DELAY_SECONDS", 0)
    chunks = [_make_chunk("some text")]
    llm = MagicMock()
    llm.invoke.side_effect = Exception("quota exceeded")
    tag_chunks_with_metadata(chunks, llm)
    assert chunks[0].metadata["topic_cluster"] == FALLBACK_METADATA["topic_cluster"]
    assert chunks[0].metadata["difficulty_tier"] == FALLBACK_METADATA["difficulty_tier"]


def test_tag_chunks_batches_correctly():
    """With batch_size=2 and 5 chunks, the LLM should be called 3 times."""
    chunks = [_make_chunk(f"chunk {i}") for i in range(5)]
    llm = MagicMock()

    def side_effect(prompt):
        # Count chunks in the prompt and return appropriately-sized response
        indices = [int(m) for m in __import__("re").findall(r"\[(\d+)\]", prompt)]
        return _make_llm_response([
            {"index": i, "topic_cluster": "LLM Fundamentals", "difficulty_tier": "beginner"}
            for i in indices
        ])

    llm.invoke.side_effect = side_effect
    tag_chunks_with_metadata(chunks, llm, batch_size=2)
    assert llm.invoke.call_count == 3  # ceil(5/2) = 3


def test_tag_chunks_returns_list():
    """tag_chunks_with_metadata should return the (mutated) list of chunks."""
    chunks = [_make_chunk("text")]
    llm = MagicMock()
    llm.invoke.return_value = _make_llm_response([
        {"index": 0, "topic_cluster": "Evaluation", "difficulty_tier": "intermediate"}
    ])
    result = tag_chunks_with_metadata(chunks, llm)
    assert result is chunks
