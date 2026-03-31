# FastAPI backend for the Grimoire AutoQuizzer application.
# Exposes REST endpoints for quiz generation and game-layer interactions
# via the RAG pipeline, replacing Streamlit as the serving layer.
# Designed to be consumed by a TypeScript frontend.
#
# Endpoints:
#   GET  /health              — liveness check
#   GET  /quiz/difficulties   — list available difficulty tiers
#   POST /quiz/generate       — generate a quiz from a topic string
#   POST /upload              — ingest documents into ChromaDB
#   POST /generate-world      — derive game zones from ChromaDB metadata
#   GET  /zones               — list unique zone names from ChromaDB
#   POST /encounter           — generate one quiz question filtered by zone
#   POST /answer              — evaluate player answer, return XP/HP delta
#
# Created: 2026-03-31
# Updated: 2026-03-31
# Author: Devon Vanaenrode

# --- Imports ---
import os
import sys
import asyncio
import logging
import shutil
import tempfile
import uuid
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
    """Parameters to generate a single combat encounter question."""
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


# Answer models
class AnswerRequest(BaseModel):
    """Player's answer to an encounter question."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {
            "question": "What does the 'R' in RAG stand for?",
            "correct_answer": "A",
            "player_answer": "A",
            "difficulty": "medium",
            "explanation": "RAG stands for Retrieval-Augmented Generation.",
        }
    })

    question: str = Field(..., min_length=1, description="The question text (echoed for UI display)")
    correct_answer: str = Field(..., min_length=1, max_length=1, description="Expected answer key, e.g. \"A\"")
    player_answer: str = Field(..., min_length=1, max_length=1, description="Player's chosen answer key, e.g. \"B\"")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty tier — determines XP/HP delta magnitude")
    explanation: str = Field(default="", description="Explanation text echoed back for UI display")


class AnswerResponse(BaseModel):
    """Result of evaluating the player's answer."""
    model_config = ConfigDict(strict=True, json_schema_extra={
        "example": {"correct": True, "xp_delta": 20, "hp_delta": 0, "difficulty": "medium", "explanation": "RAG stands for Retrieval-Augmented Generation."}
    })

    correct: bool = Field(..., description="True if player_answer matches correct_answer (case-insensitive)")
    xp_delta: int = Field(..., description="XP gained: 10/20/30 for easy/medium/hard on correct; 0 on wrong")
    hp_delta: int = Field(..., description="HP change: 0 on correct; -10/-20/-30 for easy/medium/hard on wrong")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Echoed difficulty tier")
    explanation: str = Field(..., description="Echoed explanation text for display in the UI")

# --- Helpers ---
def _build_chain_and_llm(n_questions: int, n_options: int, difficulty: str):
    """
    Initialises the LLM model, prompt template, vector store, and RAG chain.
    Returns (chain, llm_model) so both are available for invoke_with_fallback.
    get_quiz_generation_chain() discards the llm_model reference, so we call
    the lower-level functions directly here — without modifying rag_chain.py.
    Raises ValueError if the API key is missing or ChromaDB is unavailable.
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
    Runs the synchronous chain build and LLM invocation in a thread-pool
    executor so the FastAPI event loop is not blocked during the LLM call
    (which can take 10–30 seconds).
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
    Builds a RAG chain filtered to chunks belonging to the given zone (topic_cluster).
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
    No LLM call is made. Useful for populating a difficulty picker in the UI.
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
    Accepts one or more documents and runs the full ingestion pipeline:
    load → split → tag → embed → store in ChromaDB.
    Supported types: .txt, .docx, .ipynb, .py
    - 422 if any file has an unsupported extension.
    - 500 if the ingestion pipeline fails.
    Temp files are always cleaned up regardless of outcome.
    """
    # Validate all extensions before writing anything
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
    Each zone's enemy_tier is determined by the dominant difficulty_tier of its chunks.
    No LLM call is made — this is a pure metadata operation.
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


@app.post("/encounter", response_model=EncounterResponse, summary="Generate a combat encounter question")
async def encounter(request: EncounterRequest):
    """
    Generates a single quiz question using the RAG pipeline, filtered to chunks
    belonging to the requested zone (topic_cluster).
    - 404 if the zone does not exist in ChromaDB.
    - 503 if the chain cannot be initialised.
    - 500 if the LLM call fails.
    """
    # Verify the zone exists before invoking the LLM
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


@app.post("/answer", response_model=AnswerResponse, summary="Evaluate player answer")
def answer(request: AnswerRequest):
    """
    Compares the player's answer to the correct answer (case-insensitive).
    Returns XP/HP deltas based on difficulty:
      Correct → xp_delta = difficulty reward, hp_delta = 0
      Wrong   → xp_delta = 0, hp_delta = -difficulty penalty
    No LLM call or database access is made.
    """
    correct = request.player_answer.upper() == request.correct_answer.upper()
    rewards = DIFFICULTY_REWARDS[request.difficulty]

    xp_delta = rewards["xp"] if correct else 0
    hp_delta = 0 if correct else -rewards["hp_penalty"]

    return AnswerResponse(
        correct=correct,
        xp_delta=xp_delta,
        hp_delta=hp_delta,
        difficulty=request.difficulty,
        explanation=request.explanation,
    )
