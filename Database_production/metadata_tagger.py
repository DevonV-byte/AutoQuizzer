# Enriches LangChain Document chunks with game-relevant metadata by calling
# the Gemini LLM in batches. Assigns each chunk a topic_cluster (from a fixed
# taxonomy) and a difficulty_tier (beginner / intermediate / advanced).
# Also normalises the source field to a bare filename.
# These fields drive zone assignment and enemy difficulty in the Grimoire game.
#
# Created: 2026-03-27
# Author: Devon Vanaenrode

# --- Imports ---
import json
import logging
import os
import re
import time

# --- Globals ---
DEFAULT_TOPIC_TAXONOMY = [
    "LLM Fundamentals",
    "Prompt Engineering",
    "RAG Architecture",
    "LangChain",
    "Vector Databases",
    "Agents and Tools",
    "Fine-tuning",
    "Evaluation",
    "Python and Code",
    "other",
]

VALID_DIFFICULTY_TIERS = {"beginner", "intermediate", "advanced"}
FALLBACK_METADATA = {"topic_cluster": "unclassified", "difficulty_tier": "unclassified"}
DEFAULT_BATCH_SIZE = 10
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 65   # just over one minute to reset the Gemini RPM counter
CHUNK_TEXT_PREVIEW_LENGTH = 400  # chars sent to LLM per chunk (enough signal, few tokens)

logger = logging.getLogger(__name__)


# --- Helpers ---

def _build_classification_prompt(batch_chunks: list, topic_taxonomy: list) -> str:
    """
    Builds the LLM prompt for a single batch of chunks.
    Each chunk is truncated to CHUNK_TEXT_PREVIEW_LENGTH chars to keep token usage
    predictable. The LLM is instructed to return plain JSON only.
    """
    taxonomy_str = ", ".join(f'"{t}"' for t in topic_taxonomy)

    chunk_lines = "\n".join(
        f"[{i}] {chunk.page_content[:CHUNK_TEXT_PREVIEW_LENGTH]}"
        for i, chunk in enumerate(batch_chunks)
    )

    return f"""You are a metadata classifier for educational AI/ML content.
For each numbered chunk below, return ONLY a valid JSON array with no markdown fences or extra prose.
Each element must have exactly these keys:
  "index"           - integer, the chunk's 0-based index in this list
  "topic_cluster"   - string, chosen from the TAXONOMY below
  "difficulty_tier" - string, one of: "beginner", "intermediate", "advanced"

TAXONOMY: {taxonomy_str}

Rules:
- Choose the single best-fitting topic. Use "other" only if nothing fits.
- "beginner":      introductory definitions, concepts explained from scratch.
- "intermediate":  assumes prior knowledge, covers workflows or established patterns.
- "advanced":      deep internals, optimization, research-level or highly technical content.
- Output ONLY the JSON array. No explanation, no markdown, no commentary.

CHUNKS:
{chunk_lines}"""


def _parse_llm_response(response_text: str, batch_size: int) -> dict:
    """
    Parses the LLM's JSON response for a batch.
    Strips accidental markdown fences, validates structure and field values.
    Returns a dict mapping index -> {topic_cluster, difficulty_tier}.
    Returns an empty dict on any parse or validation failure so the caller can apply fallbacks.
    """
    # Strip markdown code fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?|```", "", response_text).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s\nRaw response: %s", e, response_text[:300])
        return {}

    if not isinstance(parsed, list):
        logger.warning("Expected JSON array, got %s", type(parsed).__name__)
        return {}

    result = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        topic = item.get("topic_cluster")
        difficulty = item.get("difficulty_tier")

        if not isinstance(idx, int) or not (0 <= idx < batch_size):
            logger.debug("LLM returned out-of-range index %s (batch_size=%d), skipping", idx, batch_size)
            continue
        if not isinstance(topic, str) or not topic:
            logger.warning("Missing or invalid topic_cluster at index %d", idx)
            continue
        if difficulty not in VALID_DIFFICULTY_TIERS:
            logger.warning("Invalid difficulty_tier '%s' at index %d, using fallback", difficulty, idx)
            difficulty = FALLBACK_METADATA["difficulty_tier"]

        result[idx] = {"topic_cluster": topic, "difficulty_tier": difficulty}

    return result


def _tag_batch(llm, batch_chunks: list, topic_taxonomy: list) -> dict:
    """
    Calls the LLM to classify one batch of chunks.
    Retries up to MAX_RETRIES times with exponential-ish backoff on API errors.
    Returns a parsed result dict (may be empty if all retries fail).
    """
    prompt = _build_classification_prompt(batch_chunks, topic_taxonomy)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            return _parse_llm_response(response.content, len(batch_chunks))
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, MAX_RETRIES + 1, e, RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error("LLM call failed after %d attempts: %s", MAX_RETRIES + 1, e)

    return {}


# --- Functions ---

def tag_chunks_with_metadata(
    chunks: list,
    llm,
    batch_size: int = DEFAULT_BATCH_SIZE,
    topic_taxonomy: list = DEFAULT_TOPIC_TAXONOMY,
) -> list:
    """
    Enriches each LangChain Document chunk with:
      - source:          bare filename (normalised from full path)
      - topic_cluster:   best-fit topic from topic_taxonomy, or "unclassified" on failure
      - difficulty_tier: "beginner" / "intermediate" / "advanced", or "unclassified" on failure

    Mutates chunk.metadata in place and returns the list.
    Failures on a batch are logged; affected chunks receive FALLBACK_METADATA values.
    """
    total = len(chunks)
    logger.info("Tagging %d chunks in batches of %d", total, batch_size)

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        tags = _tag_batch(llm, batch, topic_taxonomy)

        for local_idx, chunk in enumerate(batch):
            # Normalise source to bare filename
            raw_source = chunk.metadata.get("source", "unknown")
            chunk.metadata["source"] = os.path.basename(raw_source)

            # Apply LLM tags or fallback
            tag = tags.get(local_idx, FALLBACK_METADATA)
            chunk.metadata["topic_cluster"] = tag.get("topic_cluster", FALLBACK_METADATA["topic_cluster"])
            chunk.metadata["difficulty_tier"] = tag.get("difficulty_tier", FALLBACK_METADATA["difficulty_tier"])

        logger.info(
            "Tagged chunks %d-%d of %d",
            batch_start + 1, min(batch_start + batch_size, total), total,
        )

    return chunks
