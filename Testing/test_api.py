# Tests for the FastAPI backend (Backend/main.py).
# All 14 test cases covering health, difficulties, quiz generation,
# validation errors, failure modes, and CORS headers.
# LLM calls are mocked so tests run without a real API key or ChromaDB.
#
# Created: 2026-03-31
# Author: Devon Vanaenrode

# --- Imports ---
import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Ensure Code/ is on sys.path so Backend.main resolves correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backend.main import app

# --- Globals ---
VALID_QUIZ_PAYLOAD = {
    "quiz_title": "Prompt Engineering: Tier 2 – Skeleton-level",
    "difficulty": "Tier 2 – Skeleton-level",
    "questions": [
        {
            "question_number": 1,
            "question": "What is zero-shot prompting?",
            "options": {
                "A": "Providing no examples to the model",
                "B": "Providing many examples to the model",
                "C": "A type of fine-tuning",
            },
            "answer": "A",
            "explanation": "Zero-shot prompting gives the model no worked examples.",
        }
    ],
}

# --- Helpers ---
def make_client():
    """Returns a TestClient for the FastAPI app."""
    return TestClient(app)


def mock_run_quiz_generation(payload=None):
    """
    Returns an async mock for _run_quiz_generation that resolves with payload.
    Defaults to VALID_QUIZ_PAYLOAD.
    """
    import asyncio

    async def _mock(request):
        return payload if payload is not None else VALID_QUIZ_PAYLOAD

    return _mock


# --- Main loop ---
client = make_client()


# T1: Health endpoint returns 200
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# T2: Health endpoint response has version field
def test_health_response_schema():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "version" in data


# T3: Difficulties endpoint lists easy, medium, hard
def test_difficulties_returns_all_three():
    response = client.get("/quiz/difficulties")
    assert response.status_code == 200
    keys = [d["key"] for d in response.json()["difficulties"]]
    assert "easy" in keys
    assert "medium" in keys
    assert "hard" in keys


# T4: Difficulties response schema has required fields per entry
def test_difficulties_response_schema():
    response = client.get("/quiz/difficulties")
    for entry in response.json()["difficulties"]:
        assert "key" in entry
        assert "label" in entry
        assert "description" in entry


# T5: Generate quiz with valid request returns 200
def test_generate_valid_request():
    with patch("Backend.main._run_quiz_generation", new=mock_run_quiz_generation()):
        response = client.post(
            "/quiz/generate",
            json={"topic": "Prompt Engineering", "difficulty": "medium"},
        )
    assert response.status_code == 200


# T6: Generate response has required top-level fields
def test_generate_response_schema():
    with patch("Backend.main._run_quiz_generation", new=mock_run_quiz_generation()):
        response = client.post(
            "/quiz/generate",
            json={"topic": "Prompt Engineering"},
        )
    data = response.json()
    assert "quiz_title" in data
    assert "difficulty" in data
    assert "questions" in data


# T7: Each question in the response has required fields
def test_generate_questions_schema():
    with patch("Backend.main._run_quiz_generation", new=mock_run_quiz_generation()):
        response = client.post(
            "/quiz/generate",
            json={"topic": "RAG Architecture"},
        )
    questions = response.json()["questions"]
    assert len(questions) > 0
    for q in questions:
        assert "question_number" in q
        assert "question" in q
        assert "options" in q
        assert "answer" in q
        assert "explanation" in q


# T8: options field on each question is a dict with string values
def test_generate_options_is_dict():
    with patch("Backend.main._run_quiz_generation", new=mock_run_quiz_generation()):
        response = client.post(
            "/quiz/generate",
            json={"topic": "Vector Databases"},
        )
    for q in response.json()["questions"]:
        assert isinstance(q["options"], dict)
        for k, v in q["options"].items():
            assert isinstance(k, str)
            assert isinstance(v, str)


# T9: Invalid difficulty returns 422
def test_generate_invalid_difficulty():
    response = client.post(
        "/quiz/generate",
        json={"topic": "LangChain", "difficulty": "legendary"},
    )
    assert response.status_code == 422


# T10: Missing topic returns 422
def test_generate_missing_topic():
    response = client.post(
        "/quiz/generate",
        json={"difficulty": "easy"},
    )
    assert response.status_code == 422


# T11: Empty string topic returns 422
def test_generate_empty_topic():
    response = client.post(
        "/quiz/generate",
        json={"topic": ""},
    )
    assert response.status_code == 422


# T12: n_questions out of range returns 422
def test_generate_n_questions_out_of_range():
    response_low = client.post(
        "/quiz/generate",
        json={"topic": "Agents", "n_questions": 0},
    )
    response_high = client.post(
        "/quiz/generate",
        json={"topic": "Agents", "n_questions": 21},
    )
    assert response_low.status_code == 422
    assert response_high.status_code == 422


# T13: Chain init failure (ValueError) returns 503
def test_generate_chain_init_failure():
    async def _raise_value_error(request):
        raise ValueError("GOOGLE_API_KEY not found")

    with patch("Backend.main._run_quiz_generation", new=_raise_value_error):
        response = client.post(
            "/quiz/generate",
            json={"topic": "Fine-tuning"},
        )
    assert response.status_code == 503


# T14: LLM invocation failure returns 500
def test_generate_llm_exception():
    async def _raise_runtime_error(request):
        raise RuntimeError("LLM timed out")

    with patch("Backend.main._run_quiz_generation", new=_raise_runtime_error):
        response = client.post(
            "/quiz/generate",
            json={"topic": "Evaluation"},
        )
    assert response.status_code == 500


# T15 (bonus): CORS header present for TypeScript dev origin
def test_cors_headers_present():
    response = client.options(
        "/quiz/generate",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" in response.headers


# ---------------------------------------------------------------------------
# T16-T31: Game-layer endpoints
# ---------------------------------------------------------------------------

# --- Helpers for game tests ---

SAMPLE_METADATAS = [
    {"topic_cluster": "RAG Architecture", "difficulty_tier": "intermediate"},
    {"topic_cluster": "RAG Architecture", "difficulty_tier": "intermediate"},
    {"topic_cluster": "RAG Architecture", "difficulty_tier": "advanced"},
    {"topic_cluster": "LangChain", "difficulty_tier": "beginner"},
    {"topic_cluster": "LangChain", "difficulty_tier": "beginner"},
    {"topic_cluster": "LangChain", "difficulty_tier": "beginner"},
    {"topic_cluster": "LangChain", "difficulty_tier": "intermediate"},
    {"topic_cluster": "Prompt Engineering", "difficulty_tier": "beginner"},
    {"topic_cluster": "Prompt Engineering", "difficulty_tier": "beginner"},
    {"topic_cluster": "unclassified", "difficulty_tier": "unclassified"},
]

VALID_ENCOUNTER_PAYLOAD = {
    "quiz_title": "RAG Architecture: Tier 2 – Skeleton-level",
    "difficulty": "Tier 2 – Skeleton-level",
    "questions": [
        {
            "question_number": 1,
            "question": "What does RAG stand for?",
            "options": {"A": "Retrieval-Augmented Generation", "B": "Random Access Graph", "C": "Rule-based AI Gateway"},
            "answer": "A",
            "explanation": "RAG = Retrieval-Augmented Generation.",
        }
    ],
}


def make_mock_collection(metadatas=None, zone_exists=True):
    """Returns a MagicMock ChromaDB collection with configurable behaviour."""
    col = MagicMock()
    col.get.return_value = {"metadatas": metadatas if metadatas is not None else SAMPLE_METADATAS, "ids": ["id1"]}
    if not zone_exists:
        col.get.return_value = {"metadatas": [], "ids": []}
    return col


# --- Upload tests ---

# T16: valid .txt file returns 200 with chunks_added
def test_upload_valid_file_returns_200():
    with patch("Backend.main._run_ingestion", return_value=5):
        response = client.post(
            "/upload",
            files=[("files", ("notes.txt", b"some content", "text/plain"))],
        )
    assert response.status_code == 200
    assert response.json()["chunks_added"] == 5


# T17: unsupported extension returns 422
def test_upload_unsupported_extension_returns_422():
    response = client.post(
        "/upload",
        files=[("files", ("report.pdf", b"%PDF content", "application/pdf"))],
    )
    assert response.status_code == 422


# T18: no files attached returns 422
def test_upload_no_files_returns_422():
    response = client.post("/upload")
    assert response.status_code == 422


# T19: pipeline failure returns 500 and cleans up temp dir
def test_upload_pipeline_failure_returns_500():
    with patch("Backend.main._run_ingestion", side_effect=RuntimeError("disk full")):
        response = client.post(
            "/upload",
            files=[("files", ("notes.txt", b"content", "text/plain"))],
        )
    assert response.status_code == 500


# --- Generate-world tests ---

# T20: returns list of zones from ChromaDB metadata
def test_generate_world_returns_zones():
    with patch("Backend.main._get_chromadb_collection", return_value=make_mock_collection()):
        response = client.post("/generate-world")
    assert response.status_code == 200
    zones = response.json()["zones"]
    zone_names = [z["name"] for z in zones]
    assert "RAG Architecture" in zone_names
    assert "LangChain" in zone_names
    assert "Prompt Engineering" in zone_names


# T21: empty ChromaDB returns empty zone list
def test_generate_world_empty_db_returns_empty():
    mock_col = MagicMock()
    mock_col.get.return_value = {"metadatas": [], "ids": []}
    with patch("Backend.main._get_chromadb_collection", return_value=mock_col):
        response = client.post("/generate-world")
    assert response.status_code == 200
    assert response.json() == {"zones": []}


# T22: chunk counts per zone are accurate
def test_generate_world_chunk_counts_correct():
    with patch("Backend.main._get_chromadb_collection", return_value=make_mock_collection()):
        response = client.post("/generate-world")
    zones = {z["name"]: z for z in response.json()["zones"]}
    assert zones["RAG Architecture"]["chunk_count"] == 3
    assert zones["LangChain"]["chunk_count"] == 4
    assert zones["Prompt Engineering"]["chunk_count"] == 2


# --- Zones tests ---

# T23: GET /zones returns list of zone name strings
def test_get_zones_returns_names():
    with patch("Backend.main._get_chromadb_collection", return_value=make_mock_collection()):
        response = client.get("/zones")
    assert response.status_code == 200
    zones = response.json()["zones"]
    assert isinstance(zones, list)
    assert all(isinstance(z, str) for z in zones)
    assert "RAG Architecture" in zones


# T24: unclassified chunks excluded from /zones
def test_get_zones_excludes_unclassified():
    with patch("Backend.main._get_chromadb_collection", return_value=make_mock_collection()):
        response = client.get("/zones")
    assert "unclassified" not in response.json()["zones"]


# --- Encounter tests ---

# T25: valid zone + difficulty returns EncounterResponse
def test_encounter_returns_single_question():
    mock_col = make_mock_collection(zone_exists=True)
    mock_col.get.return_value = {"metadatas": SAMPLE_METADATAS, "ids": ["id1"]}

    async def _mock_encounter(request):
        return VALID_ENCOUNTER_PAYLOAD

    with patch("Backend.main._get_chromadb_collection", return_value=mock_col), \
         patch("Backend.main._build_encounter_chain", return_value=(MagicMock(), MagicMock())), \
         patch("Backend.main.invoke_with_fallback", return_value=VALID_ENCOUNTER_PAYLOAD):
        response = client.post(
            "/encounter",
            json={"zone": "RAG Architecture", "difficulty": "medium"},
        )
    assert response.status_code == 200
    data = response.json()
    for field in ["zone", "difficulty", "question", "options", "answer", "explanation"]:
        assert field in data


# T26: unknown zone returns 404
def test_encounter_unknown_zone_returns_404():
    mock_col = MagicMock()
    mock_col.get.return_value = {"metadatas": [], "ids": []}
    with patch("Backend.main._get_chromadb_collection", return_value=mock_col):
        response = client.post(
            "/encounter",
            json={"zone": "Dragons Den", "difficulty": "hard"},
        )
    assert response.status_code == 404


# T27: invalid difficulty returns 422
def test_encounter_invalid_difficulty_returns_422():
    response = client.post(
        "/encounter",
        json={"zone": "LangChain", "difficulty": "legendary"},
    )
    assert response.status_code == 422


# --- Answer tests ---

# T28: correct answer returns positive XP, zero HP delta
def test_answer_correct_returns_positive_xp():
    response = client.post(
        "/answer",
        json={
            "question": "What does RAG stand for?",
            "correct_answer": "A",
            "player_answer": "A",
            "difficulty": "easy",
            "explanation": "RAG = Retrieval-Augmented Generation.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["correct"] is True
    assert data["xp_delta"] > 0
    assert data["hp_delta"] == 0


# T29: wrong answer on hard difficulty returns max HP penalty
def test_answer_wrong_hard_returns_max_hp_penalty():
    response = client.post(
        "/answer",
        json={
            "question": "What does RAG stand for?",
            "correct_answer": "A",
            "player_answer": "C",
            "difficulty": "hard",
            "explanation": "RAG = Retrieval-Augmented Generation.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["correct"] is False
    assert data["xp_delta"] == 0
    assert data["hp_delta"] == -30


# T30: answer matching is case-insensitive
def test_answer_case_insensitive():
    response = client.post(
        "/answer",
        json={
            "question": "What does RAG stand for?",
            "correct_answer": "A",
            "player_answer": "a",
            "difficulty": "medium",
            "explanation": "",
        },
    )
    assert response.json()["correct"] is True


# T31: invalid difficulty in answer request returns 422
def test_answer_invalid_difficulty_returns_422():
    response = client.post(
        "/answer",
        json={
            "question": "What does RAG stand for?",
            "correct_answer": "A",
            "player_answer": "A",
            "difficulty": "legendary",
            "explanation": "",
        },
    )
    assert response.status_code == 422
