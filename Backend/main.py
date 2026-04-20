# FastAPI backend for the Grimoire AutoQuizzer application.
# Exposes REST endpoints for quiz generation and game-layer interactions
# via the RAG pipeline, replacing Streamlit as the serving layer.
# Designed to be consumed by a TypeScript frontend.
#
# Endpoints:
#   GET  /health                      — liveness check
#   GET  /quiz/difficulties           — list available difficulty tiers
#   POST /quiz/generate               — generate a quiz from a topic string
#   POST /upload                      — ingest documents into ChromaDB
#   POST /generate-world              — derive game zones from ChromaDB metadata
#   GET  /zones                       — list unique zone names from ChromaDB
#   POST /encounter                   — generate one quiz question filtered by zone (legacy)
#   POST /encounter/batch             — return next 5 questions from pre-generated pool
#   POST /answer                      — evaluate player answer, return XP/HP delta + progress
#   GET  /player/{player_id}/progress — per-zone/difficulty progress for a player
#
# Created: 2026-03-31
# Updated: 2026-04-20 — /encounter/batch; SQLite question pool + player progress tracking;
#                        question pool cleared on upload; file tracking with GET/DELETE /files
# Author: Devon Vanaenrode

# --- Imports ---
import os
import sys
import asyncio
import json
import logging
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from collections import Counter
from functools import partial
from typing import Literal

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Ensure Code/ directory is on sys.path so sibling packages resolve correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from RAG_Pipeline.rag_chain import (
    get_llm_model,
    create_prompt_template,
    rag_chain,
    invoke_with_fallback,
    format_docs,
    DIFFICULTY_TIERS,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)
from Database_production import embeddings
from Database_production.document_loader import load_course_documents, ALLOWED_EXTENSIONS
from Database_production.text_splitter import split_documents
from Database_production.metadata_tagger import tag_chunks_with_metadata
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser

# --- Globals ---
APP_VERSION = "1.0.0"

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:4173",
]

# XP reward and HP penalty per difficulty tier
DIFFICULTY_REWARDS = {
    "easy":   {"xp": 10, "hp_penalty": 10},
    "medium": {"xp": 20, "hp_penalty": 20},
    "hard":   {"xp": 30, "hp_penalty": 30},
}

# Maps ChromaDB difficulty_tier metadata values to in-game enemy labels
DIFFICULTY_TIER_TO_ENEMY = {
    "beginner":     "Slime",
    "intermediate": "Skeleton",
    "advanced":     "Dragon",
    "unclassified": "Slime",
}

# Question pool and quiz configuration
SQLITE_DB_PATH = os.path.join(CHROMA_DB_PATH, "grimoire.db")
POOL_SIZE = 25       # Questions pre-generated per zone+difficulty combination
QUIZ_SIZE = 5        # Questions served per quiz batch
PASS_THRESHOLD = 4   # Minimum correct to pass a quiz (out of QUIZ_SIZE)

app = FastAPI(title="Grimoire AutoQuizzer API", version=APP_VERSION)
logger = logging.getLogger(__name__)

# --- Models ---

# Quiz request/response models
class QuizRequest(BaseModel):
    """Parameters for quiz generation."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {
            "topic": "Prompt Engineering",
            "difficulty": "medium",
            "n_questions": 5,
            "n_options": 3,
        }
    })

    topic: str = Field(..., min_length=1, max_length=200, description="Topic to generate a quiz about")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium", description="Bloom's taxonomy difficulty tier: easy | medium | hard")
    n_questions: int = Field(default=5, ge=1, le=20, description="Number of questions to generate (1–20)")
    n_options: int = Field(default=3, ge=2, le=5, description="Number of answer options per question (2–5)")


class QuizQuestion(BaseModel):
    """A single multiple-choice quiz question."""
    model_config = ConfigDict(strict=True)

    question_number: int = Field(..., ge=1, description="1-based position of this question in the quiz")
    question: str = Field(..., min_length=1, description="The question text")
    options: dict[str, str] = Field(..., description="Answer options keyed by letter, e.g. {\"A\": \"...\", \"B\": \"...\"}")
    answer: str = Field(..., min_length=1, max_length=1, description="Letter key of the correct answer, e.g. \"A\"")
    explanation: str = Field(..., description="Explanation of why the correct answer is right")


class QuizResponse(BaseModel):
    """Full quiz returned by the RAG pipeline."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {
            "quiz_title": "Prompt Engineering: Tier 2 – Skeleton-level",
            "difficulty": "Tier 2 – Skeleton-level",
            "questions": [
                {
                    "question_number": 1,
                    "question": "What is zero-shot prompting?",
                    "options": {"A": "No examples given", "B": "Many examples given", "C": "Fine-tuning the model"},
                    "answer": "A",
                    "explanation": "Zero-shot prompting provides no worked examples to the model.",
                }
            ],
        }
    })

    quiz_title: str = Field(..., description="Title of the generated quiz")
    difficulty: str = Field(..., description="Full tier label, e.g. \"Tier 2 – Skeleton-level\"")
    questions: list[QuizQuestion] = Field(..., description="Ordered list of quiz questions")


class HealthResponse(BaseModel):
    """API liveness response."""
    model_config = ConfigDict(strict=True, json_schema_extra={"example": {"status": "ok", "version": "1.0.0"}})

    status: Literal["ok"] = Field(..., description="Always \"ok\" when the service is running")
    version: str = Field(..., description="API version string")


class DifficultyInfo(BaseModel):
    """Metadata for a single difficulty tier."""
    model_config = ConfigDict(strict=True)

    key: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty key used in requests")
    label: str = Field(..., description="Display label, e.g. \"Tier 1 – Slime-level\"")
    description: str = Field(..., description="Bloom's taxonomy guidance for this tier")


class DifficultiesResponse(BaseModel):
    """All available difficulty tiers."""
    model_config = ConfigDict(strict=True)

    difficulties: list[DifficultyInfo] = Field(..., description="All three difficulty tiers in order: easy, medium, hard")


# Upload models
class UploadResponse(BaseModel):
    """Result of a document ingestion run."""
    model_config = ConfigDict(strict=True, json_schema_extra={"example": {"chunks_added": 42}})

    chunks_added: int = Field(..., ge=0, description="Number of document chunks added to ChromaDB")


# World / zone models
class Zone(BaseModel):
    """A game zone derived from a topic_cluster group in ChromaDB."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {"name": "RAG Architecture", "enemy_tier": "Skeleton", "chunk_count": 12}
    })

    name: str = Field(..., description="Zone name — matches the topic_cluster metadata value in ChromaDB")
    enemy_tier: Literal["Slime", "Skeleton", "Dragon"] = Field(..., description="Enemy type derived from dominant difficulty_tier: Slime (beginner) | Skeleton (intermediate) | Dragon (advanced)")
    chunk_count: int = Field(..., ge=1, description="Number of document chunks belonging to this zone")


class GenerateWorldResponse(BaseModel):
    """All zones generated from ChromaDB metadata."""
    model_config = ConfigDict(strict=True)

    zones: list[Zone] = Field(..., description="All game zones sorted alphabetically by name")


class ZonesResponse(BaseModel):
    """Lightweight list of zone names currently in ChromaDB."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {"zones": ["Agents and Tools", "LangChain", "RAG Architecture"]}
    })

    zones: list[str] = Field(..., description="Sorted list of zone names (topic_cluster values), excluding 'unclassified'")


# Encounter models
class EncounterRequest(BaseModel):
    """Parameters to generate a single combat encounter question (legacy)."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {"zone": "RAG Architecture", "difficulty": "medium"}
    })

    zone: str = Field(..., min_length=1, max_length=100, description="Zone name (topic_cluster) to filter ChromaDB retrieval")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium", description="Bloom's taxonomy difficulty tier")


class EncounterResponse(BaseModel):
    """A single combat encounter question."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {
            "zone": "RAG Architecture",
            "difficulty": "medium",
            "question": "What does the 'R' in RAG stand for?",
            "options": {"A": "Retrieval", "B": "Recursive", "C": "Relational"},
            "answer": "A",
            "explanation": "RAG stands for Retrieval-Augmented Generation.",
        }
    })

    zone: str = Field(..., description="Zone this question was generated for")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty tier of this encounter")
    question: str = Field(..., description="The question text")
    options: dict[str, str] = Field(..., description="Answer options keyed by letter, e.g. {\"A\": \"...\", \"B\": \"...\"}")
    answer: str = Field(..., min_length=1, max_length=1, description="Letter key of the correct answer")
    explanation: str = Field(..., description="Explanation of the correct answer")


class EncounterBatchRequest(BaseModel):
    """Parameters to fetch the next quiz batch from the pre-generated question pool."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {"player_id": "abc123", "zone": "RAG Architecture", "difficulty": "medium"}
    })

    player_id: str = Field(..., min_length=1, max_length=100, description="Unique player identifier")
    zone: str = Field(..., min_length=1, max_length=100, description="Zone name (topic_cluster)")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="medium", description="Bloom's taxonomy difficulty tier")


class EncounterBatchResponse(BaseModel):
    """A batch of QUIZ_SIZE questions from the pre-generated pool."""
    model_config = ConfigDict(strict=True)

    zone: str = Field(..., description="Zone these questions were generated for")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty tier")
    quiz_number: int = Field(..., ge=1, description="Which quiz batch this is (1 = first 5, 2 = second 5, …)")
    questions: list[EncounterResponse] = Field(..., description=f"Exactly {QUIZ_SIZE} questions from the pool")


# Answer models
class AnswerRequest(BaseModel):
    """Player's answer to an encounter question."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {
            "player_id": "abc123",
            "zone": "RAG Architecture",
            "question": "What does the 'R' in RAG stand for?",
            "correct_answer": "A",
            "player_answer": "A",
            "difficulty": "medium",
            "explanation": "RAG stands for Retrieval-Augmented Generation.",
        }
    })

    player_id: str = Field(..., min_length=1, max_length=100, description="Unique player identifier")
    zone: str = Field(..., min_length=1, max_length=100, description="Zone name for progress tracking")
    question: str = Field(..., min_length=1, description="The question text (echoed for UI display)")
    correct_answer: str = Field(..., min_length=1, max_length=1, description="Expected answer key, e.g. \"A\"")
    player_answer: str = Field(..., min_length=1, max_length=1, description="Player's chosen answer key, e.g. \"B\"")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty tier — determines XP/HP delta magnitude")
    explanation: str = Field(default="", description="Explanation text echoed back for UI display")


class AnswerResponse(BaseModel):
    """Result of evaluating the player's answer."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {
            "correct": True,
            "xp_delta": 20,
            "hp_delta": 0,
            "difficulty": "medium",
            "explanation": "RAG stands for Retrieval-Augmented Generation.",
            "quiz_complete": False,
            "quiz_score": None,
            "level_complete": False,
        }
    })

    correct: bool = Field(..., description="True if player_answer matches correct_answer (case-insensitive)")
    xp_delta: int = Field(..., description="XP gained: 10/20/30 for easy/medium/hard on correct; 0 on wrong")
    hp_delta: int = Field(..., description="HP change: 0 on correct; -10/-20/-30 for easy/medium/hard on wrong")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Echoed difficulty tier")
    explanation: str = Field(..., description="Echoed explanation text for display in the UI")
    quiz_complete: bool = Field(..., description=f"True when the {QUIZ_SIZE}th answer of a quiz batch is recorded")
    quiz_score: int | None = Field(..., description=f"Correct answers in the completed quiz (0–{QUIZ_SIZE}); null if not yet complete")
    level_complete: bool = Field(..., description=f"True when all {POOL_SIZE // QUIZ_SIZE} quiz batches are passed at >= {PASS_THRESHOLD}/{QUIZ_SIZE}")


# Player progress models
class PlayerProgressEntry(BaseModel):
    """Progress for one zone+difficulty combination."""
    model_config = ConfigDict(strict=True)

    zone: str = Field(..., description="Zone name")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty tier")
    questions_answered: int = Field(..., ge=0, description="Total questions answered in this zone+difficulty")
    correct_count: int = Field(..., ge=0, description="Total correct answers")
    quizzes_passed: int = Field(..., ge=0, description=f"Quiz batches passed at >= {PASS_THRESHOLD}/{QUIZ_SIZE}")
    level_complete: bool = Field(..., description="Whether all quiz batches for this level have been completed")


class PlayerProgressResponse(BaseModel):
    """All progress for a player across all zones and difficulties."""
    model_config = ConfigDict(strict=True)

    player_id: str = Field(..., description="The player's unique identifier")
    progress: list[PlayerProgressEntry] = Field(..., description="Per-zone/difficulty progress entries")


# File tracking models
class FileEntry(BaseModel):
    """A single file that has been ingested into ChromaDB."""
    model_config = ConfigDict(strict=True)

    filename: str = Field(..., description="Bare filename as stored in ChromaDB chunk metadata")
    uploaded_at: str = Field(..., description="ISO 8601 UTC timestamp of when the file was uploaded")


class FilesResponse(BaseModel):
    """All files currently tracked in the database."""
    model_config = ConfigDict(strict=True)

    files: list[FileEntry] = Field(..., description="Files sorted by upload time, most recent first")


class MessageResponse(BaseModel):
    """Generic success message."""
    model_config = ConfigDict(strict=True)

    message: str = Field(..., description="Human-readable result description")


# --- Helpers ---

# DB helpers

def init_db() -> None:
    """
    Creates SQLite tables for question pool and player progress if they don't exist.
    Safe to call on every startup (uses IF NOT EXISTS).
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS question_pool (
                zone            TEXT    NOT NULL,
                difficulty      TEXT    NOT NULL,
                question_index  INTEGER NOT NULL,
                question_json   TEXT    NOT NULL,
                PRIMARY KEY (zone, difficulty, question_index)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                filename    TEXT PRIMARY KEY,
                uploaded_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS player_progress (
                player_id             TEXT    NOT NULL,
                zone                  TEXT    NOT NULL,
                difficulty            TEXT    NOT NULL,
                questions_served      INTEGER NOT NULL DEFAULT 0,
                questions_answered    INTEGER NOT NULL DEFAULT 0,
                correct_count         INTEGER NOT NULL DEFAULT 0,
                current_quiz_correct  INTEGER NOT NULL DEFAULT 0,
                quiz_scores           TEXT    NOT NULL DEFAULT '[]',
                level_complete        INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (player_id, zone, difficulty)
            )
        """)


def _record_uploaded_files(filenames: list[str]) -> None:
    """Upserts filenames into uploaded_files with the current UTC timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO uploaded_files (filename, uploaded_at) VALUES (?,?)",
            [(f, now) for f in filenames],
        )


def _delete_chromadb_file(filename: str) -> None:
    """
    Deletes all ChromaDB chunks whose source metadata matches filename.
    No-op if the collection does not exist.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        return
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        collection.delete(where={"source": {"$eq": filename}})
    except Exception:
        pass  # collection may not exist yet


def _clear_all_databases() -> None:
    """
    Wipes all state: ChromaDB collection, question pool, player progress, and uploaded files.
    Called when the user requests a full database reset.
    """
    # Drop and let ChromaDB recreate the collection on next upload
    if os.path.exists(CHROMA_DB_PATH):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute("DELETE FROM question_pool")
        conn.execute("DELETE FROM player_progress")
        conn.execute("DELETE FROM uploaded_files")


def _build_pool_chain(zone: str, difficulty: str):
    """
    Builds a RAG chain configured to generate POOL_SIZE questions in one LLM call.
    Filters retrieval to chunks belonging to the given zone (topic_cluster).
    Returns (chain, llm_model) for use with invoke_with_fallback.
    """
    llm_model = get_llm_model()
    prompt = create_prompt_template(n_questions=POOL_SIZE, n_options=3, difficulty=difficulty)
    embeddings_model = embeddings.get_embeddings_model()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
    )
    _, _, k = DIFFICULTY_TIERS[difficulty]
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"topic_cluster": {"$eq": zone}},
        }
    )
    chain = (
        {
            "context": (lambda x: x["topic"]) | retriever | format_docs,
            "topic": lambda x: x["topic"],
        }
        | prompt
        | llm_model
        | JsonOutputParser()
    )
    return chain, llm_model


def _generate_and_store_pool(zone: str, difficulty: str) -> None:
    """
    Generates POOL_SIZE questions for (zone, difficulty) via the LLM and stores them in SQLite.
    Replaces any existing pool for this combination.
    Raises ValueError if the LLM returns fewer than POOL_SIZE questions.
    """
    chain, llm_model = _build_pool_chain(zone, difficulty)
    quiz_data = invoke_with_fallback(chain, llm_model, zone)
    questions = quiz_data.get("questions", [])

    if len(questions) < POOL_SIZE:
        raise ValueError(
            f"LLM returned only {len(questions)} questions for '{zone}'/{difficulty}, expected {POOL_SIZE}."
        )

    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute(
            "DELETE FROM question_pool WHERE zone=? AND difficulty=?",
            (zone, difficulty),
        )
        conn.executemany(
            "INSERT INTO question_pool (zone, difficulty, question_index, question_json) VALUES (?,?,?,?)",
            [(zone, difficulty, i, json.dumps(q)) for i, q in enumerate(questions[:POOL_SIZE])],
        )


def _get_or_create_batch(player_id: str, zone: str, difficulty: str) -> tuple[list[dict], int]:
    """
    Returns (questions, quiz_number) for the next QUIZ_SIZE questions in the pool.
    Generates the pool on first access; regenerates it when exhausted.
    Advances the player's questions_served pointer by QUIZ_SIZE.
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        progress_row = conn.execute(
            "SELECT questions_served FROM player_progress WHERE player_id=? AND zone=? AND difficulty=?",
            (player_id, zone, difficulty),
        ).fetchone()
        served = progress_row[0] if progress_row else 0

        pool_count = conn.execute(
            "SELECT COUNT(*) FROM question_pool WHERE zone=? AND difficulty=?",
            (zone, difficulty),
        ).fetchone()[0]

    # Generate pool if missing or the player has exhausted it
    if pool_count < POOL_SIZE or served >= POOL_SIZE:
        _generate_and_store_pool(zone, difficulty)
        served = 0
        if progress_row:
            with sqlite3.connect(SQLITE_DB_PATH) as conn:
                conn.execute(
                    "UPDATE player_progress SET questions_served=0 WHERE player_id=? AND zone=? AND difficulty=?",
                    (player_id, zone, difficulty),
                )

    # Fetch the next QUIZ_SIZE questions and advance the served pointer
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT question_json FROM question_pool "
            "WHERE zone=? AND difficulty=? AND question_index >= ? "
            "ORDER BY question_index LIMIT ?",
            (zone, difficulty, served, QUIZ_SIZE),
        ).fetchall()
        questions = [json.loads(r[0]) for r in rows]
        quiz_number = (served // QUIZ_SIZE) + 1

        new_served = served + QUIZ_SIZE
        if progress_row:
            conn.execute(
                "UPDATE player_progress SET questions_served=? WHERE player_id=? AND zone=? AND difficulty=?",
                (new_served, player_id, zone, difficulty),
            )
        else:
            conn.execute(
                "INSERT INTO player_progress (player_id, zone, difficulty, questions_served) VALUES (?,?,?,?)",
                (player_id, zone, difficulty, new_served),
            )

    return questions, quiz_number


def _record_answer(player_id: str, zone: str, difficulty: str, correct: bool) -> dict:
    """
    Persists the result of a single answer for (player_id, zone, difficulty).
    Returns:
        quiz_complete  (bool)     — True when QUIZ_SIZE answers are recorded for the current batch
        quiz_score     (int|None) — Correct answers in the just-completed quiz; None if not yet done
        level_complete (bool)     — True if all POOL_SIZE//QUIZ_SIZE batches each hit >= PASS_THRESHOLD
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT questions_answered, correct_count, current_quiz_correct, quiz_scores, level_complete "
            "FROM player_progress WHERE player_id=? AND zone=? AND difficulty=?",
            (player_id, zone, difficulty),
        ).fetchone()

        if row:
            questions_answered, correct_count, current_quiz_correct, quiz_scores_json, level_complete_int = row
            quiz_scores: list[int] = json.loads(quiz_scores_json)
        else:
            questions_answered = correct_count = current_quiz_correct = 0
            quiz_scores = []
            level_complete_int = 0

        questions_answered += 1
        if correct:
            correct_count += 1
            current_quiz_correct += 1

        quiz_complete = False
        quiz_score = None
        level_complete = bool(level_complete_int)

        # Check if this answer completes a quiz batch
        if questions_answered % QUIZ_SIZE == 0:
            quiz_complete = True
            quiz_score = current_quiz_correct
            quiz_scores.append(current_quiz_correct)
            current_quiz_correct = 0

            # Level complete when the last POOL_SIZE//QUIZ_SIZE batches all pass
            n_quizzes = POOL_SIZE // QUIZ_SIZE
            if len(quiz_scores) >= n_quizzes:
                recent = quiz_scores[-n_quizzes:]
                if all(s >= PASS_THRESHOLD for s in recent):
                    level_complete = True

        if row:
            conn.execute(
                "UPDATE player_progress SET "
                "questions_answered=?, correct_count=?, current_quiz_correct=?, quiz_scores=?, level_complete=? "
                "WHERE player_id=? AND zone=? AND difficulty=?",
                (questions_answered, correct_count, current_quiz_correct,
                 json.dumps(quiz_scores), int(level_complete),
                 player_id, zone, difficulty),
            )
        else:
            conn.execute(
                "INSERT INTO player_progress "
                "(player_id, zone, difficulty, questions_served, questions_answered, correct_count, "
                "current_quiz_correct, quiz_scores, level_complete) "
                "VALUES (?,?,?,0,?,?,?,?,?)",
                (player_id, zone, difficulty,
                 questions_answered, correct_count, current_quiz_correct,
                 json.dumps(quiz_scores), int(level_complete)),
            )

    return {"quiz_complete": quiz_complete, "quiz_score": quiz_score, "level_complete": level_complete}


# Chain / LLM helpers

def _build_chain_and_llm(n_questions: int, n_options: int, difficulty: str):
    """
    Initialises the LLM model, prompt template, vector store, and RAG chain.
    Returns (chain, llm_model) so both are available for invoke_with_fallback.
    """
    llm_model = get_llm_model()
    prompt = create_prompt_template(n_questions, n_options, difficulty)
    embeddings_model = embeddings.get_embeddings_model()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
    )
    chain = rag_chain(llm_model, prompt, vectorstore, difficulty)
    return chain, llm_model


async def _run_quiz_generation(request: QuizRequest) -> dict:
    """
    Runs the synchronous chain build and LLM invocation in a thread-pool executor
    so the FastAPI event loop is not blocked during the LLM call.
    """
    loop = asyncio.get_event_loop()
    chain, llm_model = await loop.run_in_executor(
        None,
        partial(_build_chain_and_llm, request.n_questions, request.n_options, request.difficulty),
    )
    result = await loop.run_in_executor(
        None,
        partial(invoke_with_fallback, chain, llm_model, request.topic),
    )
    return result


def _get_chromadb_collection():
    """
    Opens the ChromaDB persistent client and returns the autoquizzer collection.
    Raises FileNotFoundError if the database path does not exist.
    Raises ValueError if the collection has not been created yet.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB path not found: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)


def _derive_zones(metadatas: list[dict]) -> list[Zone]:
    """
    Pure function — derives Zone objects from a flat list of ChromaDB metadata dicts.
    Groups by topic_cluster, excludes 'unclassified', and picks the dominant
    difficulty_tier per group (higher tier wins on tie: Dragon > Skeleton > Slime).
    """
    tier_rank = {"advanced": 3, "intermediate": 2, "beginner": 1, "unclassified": 0}

    groups: dict[str, list[str]] = {}
    for meta in metadatas:
        cluster = meta.get("topic_cluster", "unclassified")
        if cluster == "unclassified":
            continue
        groups.setdefault(cluster, []).append(meta.get("difficulty_tier", "unclassified"))

    zones = []
    for name, tiers in groups.items():
        counts = Counter(tiers)
        dominant = max(counts, key=lambda t: (counts[t], tier_rank.get(t, 0)))
        zones.append(Zone(
            name=name,
            enemy_tier=DIFFICULTY_TIER_TO_ENEMY.get(dominant, "Slime"),
            chunk_count=len(tiers),
        ))

    return sorted(zones, key=lambda z: z.name)


def _build_encounter_chain(zone: str, difficulty: str):
    """
    Builds a RAG chain filtered to chunks belonging to the given zone.
    Forces n_questions=1 so the LLM returns a single encounter question.
    Returns (chain, llm_model) for use with invoke_with_fallback.
    """
    llm_model = get_llm_model()
    prompt = create_prompt_template(n_questions=1, n_options=3, difficulty=difficulty)
    embeddings_model = embeddings.get_embeddings_model()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
    )
    _, _, k = DIFFICULTY_TIERS[difficulty]
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"topic_cluster": {"$eq": zone}},
        }
    )
    chain = (
        {
            "context": (lambda x: x["topic"]) | retriever | format_docs,
            "topic": lambda x: x["topic"],
        }
        | prompt
        | llm_model
        | JsonOutputParser()
    )
    return chain, llm_model


def _clear_question_pool() -> None:
    """
    Deletes all pre-generated question pools from SQLite so they regenerate
    lazily on next access. Called after any document upload to prevent stale questions.
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute("DELETE FROM question_pool")


def _run_ingestion(temp_dir: str) -> int:
    """
    Synchronous ingestion pipeline: load → split → tag → embed → store.
    Runs in a thread-pool executor so it doesn't block the event loop.
    Returns the number of chunks added to ChromaDB.
    """
    llm_model = get_llm_model()
    documents = load_course_documents(temp_dir)
    chunks = split_documents(documents)
    tagged_chunks = tag_chunks_with_metadata(chunks, llm_model)

    embeddings_model = embeddings.get_embeddings_model()
    ids = [uuid.uuid4().hex for _ in tagged_chunks]

    Chroma.from_documents(
        documents=tagged_chunks,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        ids=ids,
    )
    return len(tagged_chunks)


# Initialise SQLite tables on module load
init_db()

# --- Main loop ---

# CORS — allow TypeScript dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health", response_model=HealthResponse, summary="Liveness check")
def health():
    """Returns 200 OK when the service is running."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/quiz/difficulties", response_model=DifficultiesResponse, summary="List difficulty tiers")
def get_difficulties():
    """
    Returns all available quiz difficulty tiers derived from DIFFICULTY_TIERS.
    No LLM call is made.
    """
    difficulties = [
        DifficultyInfo(key=key, label=label, description=instructions)
        for key, (label, instructions, _) in DIFFICULTY_TIERS.items()
    ]
    return DifficultiesResponse(difficulties=difficulties)


@app.post("/quiz/generate", response_model=QuizResponse, summary="Generate a quiz")
async def generate_quiz(request: QuizRequest):
    """
    Generates a multiple-choice quiz on the given topic using the RAG pipeline.
    - 503 if the chain cannot be initialised (missing API key, ChromaDB unavailable).
    - 500 if the LLM call or JSON parsing fails after the fallback retry.
    """
    try:
        quiz_data = await _run_quiz_generation(request)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Chain initialisation failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except Exception as e:
        logger.error("Quiz generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {e}")

    if not isinstance(quiz_data, dict) or "questions" not in quiz_data:
        raise HTTPException(status_code=500, detail="LLM returned an unexpected response format.")

    return QuizResponse(**quiz_data)


@app.post("/upload", response_model=UploadResponse, summary="Ingest documents into ChromaDB")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Accepts one or more documents and runs the full ingestion pipeline.
    Supported types: .txt, .docx, .ipynb, .py
    - 422 if any file has an unsupported extension.
    - 500 if the ingestion pipeline fails.
    Temp files are always cleaned up regardless of outcome.
    """
    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

    temp_dir = tempfile.mkdtemp()
    try:
        for file in files:
            dest = os.path.join(temp_dir, file.filename)
            contents = await file.read()
            with open(dest, "wb") as f:
                f.write(contents)

        loop = asyncio.get_event_loop()
        chunks_added = await loop.run_in_executor(None, partial(_run_ingestion, temp_dir))
        await loop.run_in_executor(None, _clear_question_pool)
        filenames = [f.filename for f in files if f.filename]
        await loop.run_in_executor(None, partial(_record_uploaded_files, filenames))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return UploadResponse(chunks_added=chunks_added)


@app.post("/generate-world", response_model=GenerateWorldResponse, summary="Generate game world from ChromaDB")
async def generate_world():
    """
    Reads ChromaDB metadata and groups chunks by topic_cluster into game zones.
    No LLM call is made.
    - 503 if ChromaDB is unavailable.
    """
    try:
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, _get_chromadb_collection)
        result = await loop.run_in_executor(
            None, lambda: collection.get(include=["metadatas"])
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    zones = _derive_zones(result.get("metadatas") or [])
    return GenerateWorldResponse(zones=zones)


@app.get("/zones", response_model=ZonesResponse, summary="List available zones")
async def get_zones():
    """
    Returns a sorted list of unique zone names (topic_cluster values) in ChromaDB.
    Excludes 'unclassified'. No LLM call is made.
    - 503 if ChromaDB is unavailable.
    """
    try:
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, _get_chromadb_collection)
        result = await loop.run_in_executor(
            None, lambda: collection.get(include=["metadatas"])
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    metadatas = result.get("metadatas") or []
    zone_names = sorted({
        m.get("topic_cluster", "unclassified")
        for m in metadatas
        if m.get("topic_cluster", "unclassified") != "unclassified"
    })
    return ZonesResponse(zones=zone_names)


@app.post("/encounter", response_model=EncounterResponse, summary="Generate a single encounter question (legacy)")
async def encounter(request: EncounterRequest):
    """
    Generates a single quiz question via the RAG pipeline, filtered to the requested zone.
    Legacy endpoint — prefer /encounter/batch for production use.
    - 404 if the zone does not exist in ChromaDB.
    - 503 if the chain cannot be initialised.
    - 500 if the LLM call fails.
    """
    try:
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, _get_chromadb_collection)
        check = await loop.run_in_executor(
            None,
            lambda: collection.get(
                where={"topic_cluster": {"$eq": request.zone}},
                limit=1,
            ),
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    if not check.get("ids"):
        raise HTTPException(status_code=404, detail=f"Zone '{request.zone}' not found in ChromaDB.")

    try:
        chain, llm_model = await loop.run_in_executor(
            None,
            partial(_build_encounter_chain, request.zone, request.difficulty),
        )
        quiz_data = await loop.run_in_executor(
            None,
            partial(invoke_with_fallback, chain, llm_model, request.zone),
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except Exception as e:
        logger.error("Encounter generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Encounter generation failed: {e}")

    if not isinstance(quiz_data, dict) or not quiz_data.get("questions"):
        raise HTTPException(status_code=500, detail="LLM returned an unexpected response format.")

    q = quiz_data["questions"][0]
    return EncounterResponse(
        zone=request.zone,
        difficulty=request.difficulty,
        question=q["question"],
        options=q["options"],
        answer=q["answer"],
        explanation=q["explanation"],
    )


@app.post("/encounter/batch", response_model=EncounterBatchResponse, summary="Fetch next quiz batch from pre-generated pool")
async def encounter_batch(request: EncounterBatchRequest):
    """
    Returns QUIZ_SIZE questions from the pre-generated pool for (zone, difficulty).
    On first access per (zone, difficulty) pair, generates the full POOL_SIZE question pool
    via the LLM — this may take up to 60 seconds. Subsequent calls are instant (DB reads).
    The pool regenerates automatically when the player exhausts all POOL_SIZE questions.
    - 404 if the zone does not exist in ChromaDB.
    - 503 if pool generation fails.
    - 500 if the LLM call fails.
    """
    try:
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, _get_chromadb_collection)
        check = await loop.run_in_executor(
            None,
            lambda: collection.get(
                where={"topic_cluster": {"$eq": request.zone}},
                limit=1,
            ),
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")

    if not check.get("ids"):
        raise HTTPException(status_code=404, detail=f"Zone '{request.zone}' not found in ChromaDB.")

    try:
        questions, quiz_number = await loop.run_in_executor(
            None,
            partial(_get_or_create_batch, request.player_id, request.zone, request.difficulty),
        )
    except (ValueError, FileNotFoundError) as e:
        logger.error("Pool generation failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except Exception as e:
        logger.error("Batch fetch failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch fetch failed: {e}")

    encounter_questions = [
        EncounterResponse(
            zone=request.zone,
            difficulty=request.difficulty,
            question=q["question"],
            options=q["options"],
            answer=q["answer"],
            explanation=q["explanation"],
        )
        for q in questions
    ]
    return EncounterBatchResponse(
        zone=request.zone,
        difficulty=request.difficulty,
        quiz_number=quiz_number,
        questions=encounter_questions,
    )


@app.post("/answer", response_model=AnswerResponse, summary="Evaluate player answer")
def answer(request: AnswerRequest):
    """
    Compares the player's answer to the correct answer (case-insensitive).
    Records the result in player_progress and returns:
      - XP/HP deltas based on difficulty
      - quiz_complete / quiz_score when the QUIZ_SIZE-th answer in a batch is recorded
      - level_complete when all POOL_SIZE // QUIZ_SIZE batches are passed at >= PASS_THRESHOLD
    No LLM call is made.
    """
    correct = request.player_answer.upper() == request.correct_answer.upper()
    rewards = DIFFICULTY_REWARDS[request.difficulty]

    xp_delta = rewards["xp"] if correct else 0
    hp_delta = 0 if correct else -rewards["hp_penalty"]

    progress = _record_answer(request.player_id, request.zone, request.difficulty, correct)

    return AnswerResponse(
        correct=correct,
        xp_delta=xp_delta,
        hp_delta=hp_delta,
        difficulty=request.difficulty,
        explanation=request.explanation,
        quiz_complete=progress["quiz_complete"],
        quiz_score=progress["quiz_score"],
        level_complete=progress["level_complete"],
    )


@app.get("/player/{player_id}/progress", response_model=PlayerProgressResponse, summary="Get player progress")
def get_player_progress(player_id: str):
    """
    Returns per-zone/difficulty progress for the given player.
    No LLM call or ChromaDB access is made.
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT zone, difficulty, questions_answered, correct_count, quiz_scores, level_complete "
            "FROM player_progress WHERE player_id=?",
            (player_id,),
        ).fetchall()

    progress = [
        PlayerProgressEntry(
            zone=row[0],
            difficulty=row[1],
            questions_answered=row[2],
            correct_count=row[3],
            quizzes_passed=sum(1 for s in json.loads(row[4]) if s >= PASS_THRESHOLD),
            level_complete=bool(row[5]),
        )
        for row in rows
    ]
    return PlayerProgressResponse(player_id=player_id, progress=progress)


@app.get("/files", response_model=FilesResponse, summary="List uploaded files")
def get_files():
    """
    Returns all filenames currently tracked in the database, most recent first.
    No LLM call or ChromaDB access is made.
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT filename, uploaded_at FROM uploaded_files ORDER BY uploaded_at DESC"
        ).fetchall()
    return FilesResponse(files=[FileEntry(filename=r[0], uploaded_at=r[1]) for r in rows])


@app.delete("/files", response_model=MessageResponse, summary="Clear entire database")
async def clear_all_files():
    """
    Wipes all state: ChromaDB collection, question pool, player progress, and uploaded files.
    This is irreversible.
    - 500 if the wipe fails.
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _clear_all_databases)
    except Exception as e:
        logger.error("Clear all failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Clear failed: {e}")
    return MessageResponse(message="Database cleared.")


@app.delete("/files/{filename}", response_model=MessageResponse, summary="Delete a specific file")
async def delete_file(filename: str):
    """
    Deletes all ChromaDB chunks for the given filename, removes it from uploaded_files,
    and clears the question pool (so questions regenerate without the deleted content).
    Player progress is preserved.
    - 404 if the filename is not tracked.
    - 500 if deletion fails.
    """
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        row = conn.execute(
            "SELECT filename FROM uploaded_files WHERE filename=?", (filename,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(_delete_chromadb_file, filename))
        await loop.run_in_executor(None, _clear_question_pool)
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.execute("DELETE FROM uploaded_files WHERE filename=?", (filename,))
    except Exception as e:
        logger.error("File deletion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Deletion failed: {e}")

    return MessageResponse(message=f"'{filename}' deleted.")
