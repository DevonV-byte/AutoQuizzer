# Create the core RAG chain using LCEL. Supports three Bloom's Taxonomy difficulty tiers:
# Tier 1 (Slime-level): recall and recognition; Tier 2 (Skeleton-level): understanding
# and application; Tier 3 (Dragon-level): cross-topic synthesis and analysis.
# Includes invoke_with_fallback() which re-prompts once with a stricter JSON instruction
# if the initial LLM response fails to parse.
#
# Created: 2026-03-20
# Author: Devon Vanaenrode
# Updated: 2026-03-30
# --- Imports ---
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
import json

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage

from Database_production.document_loader import COURSE_DIR, load_course_documents
from Database_production.text_splitter import split_documents
from Database_production import embeddings

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "..", "Database")
COLLECTION_NAME = "autoquizzer_collection"

# Maps difficulty string → (game tier label, Bloom's taxonomy instruction, retriever k)
DIFFICULTY_TIERS = {
    "easy": (
        "Tier 1 – Slime-level",
        "Focus on recall and recognition. Ask about definitions, facts, and direct "
        "identification of concepts from the material. Questions should test whether "
        "the student can remember and recognise key terms.",
        2,
    ),
    "medium": (
        "Tier 2 – Skeleton-level",
        "Focus on conceptual understanding and application. Ask questions that require "
        "the student to explain ideas in their own words, compare concepts, or apply a "
        "principle to a given scenario.",
        4,
    ),
    "hard": (
        "Tier 3 – Dragon-level",
        "Focus on cross-topic synthesis and analysis. Draw on multiple concepts from "
        "the provided context and ask questions that require the student to integrate "
        "information, identify relationships between topics, evaluate trade-offs, or "
        "construct an argument.",
        8,
    ),
}

# --- Functions ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm_model():
    """
    Initializes and returns the Chat Google Generative AI model.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", google_api_key=api_key)

def create_prompt_template(n_questions=5, n_options=3, difficulty="medium"):
    """
    Builds a PromptTemplate for quiz generation.
    difficulty must be one of: 'easy', 'medium', 'hard'.
    Each tier maps to a Bloom's Taxonomy instruction set injected into the prompt.
    """
    if difficulty not in DIFFICULTY_TIERS:
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. Must be one of: {list(DIFFICULTY_TIERS.keys())}"
        )

    tier_label, tier_instructions, _ = DIFFICULTY_TIERS[difficulty]

    quiz_template = """You are a quiz master. Generate a quiz based only on the provided context.

Number of questions: {n_questions}
Number of options per question: {n_options}
Difficulty tier: {tier_label}
Bloom's taxonomy guidance: {tier_instructions}
Topic: {topic}

Context:
{context}

Return ONLY a valid JSON object. No markdown, no code fences, no explanatory text before or after.
Use this exact structure:
{{
    "quiz_title": "...",
    "difficulty": "{tier_label}",
    "questions": [
        {{
            "question_number": 1,
            "question": "...",
            "options": {{"A": "...", "B": "...", "C": "..."}},
            "answer": "A",
            "explanation": "..."
        }}
    ]
}}
"""

    prompt_template = PromptTemplate(
        template=quiz_template,
        input_variables=["context", "topic"],
        partial_variables={
            "n_questions": n_questions,
            "n_options": n_options,
            "tier_label": tier_label,
            "tier_instructions": tier_instructions,
        }
    )

    return prompt_template

def rag_chain(llm_model, prompt_template, vectorstore, difficulty="medium"):
    """
    Builds and returns the LCEL chain.
    Retriever k is set from DIFFICULTY_TIERS so Dragon-level pulls more chunks for synthesis.
    Input must be a dict with keys: "topic"
    """
    _, _, k = DIFFICULTY_TIERS.get(difficulty, DIFFICULTY_TIERS["medium"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    qa_chain = (
        {
            "context": (lambda x: x["topic"]) | retriever | format_docs,
            "topic": lambda x: x["topic"],
        }
        | prompt_template
        | llm_model
        | JsonOutputParser()
    )

    return qa_chain

FALLBACK_PROMPT = """Your previous response could not be parsed as JSON.
Return ONLY a raw JSON object — no markdown, no code fences, no explanatory text.
The object must follow this exact structure:
{{
    "quiz_title": "...",
    "difficulty": "...",
    "questions": [
        {{
            "question_number": 1,
            "question": "...",
            "options": {{"A": "...", "B": "...", "C": "..."}},
            "answer": "A",
            "explanation": "..."
        }}
    ]
}}
Previous (invalid) response:
{invalid_response}
"""

def invoke_with_fallback(chain, llm_model, topic):
    """
    Invokes the chain with the given topic.
    If the response fails to parse as JSON, re-prompts the LLM once with a stricter
    format instruction. Raises on the second failure.
    """
    try:
        return chain.invoke({"topic": topic})
    except (OutputParserException, json.JSONDecodeError, ValueError) as first_error:
        print(f"Initial parse failed ({first_error}). Retrying with stricter prompt...")
        invalid_response = str(first_error)
        retry_message = HumanMessage(content=FALLBACK_PROMPT.format(invalid_response=invalid_response))
        raw = llm_model.invoke([retry_message])
        return JsonOutputParser().parse(raw.content)

def get_quiz_generation_chain(n_questions=3, n_options=3, difficulty="medium"):
    """
    Initializes and returns the quiz generation RAG chain.
    Accepts n_questions, n_options, and difficulty to configure the prompt template.
    """
    try:
        llm_model = get_llm_model()
        prompt = create_prompt_template(n_questions, n_options, difficulty)
        embeddings_model = embeddings.get_embeddings_model()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings_model
        )
        return rag_chain(llm_model, prompt, vectorstore, difficulty)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error initializing quiz generation chain: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during chain initialization: {e}")
        return None

# --- Main Execution ---
def main():
    """
    Create the LLM model, connect ChromaDB to the prompt, pipe it into Gemini and test with simple string.
    """
    RAG_chain = get_quiz_generation_chain()

    if RAG_chain:
        # Test our retrievalQA
        quiz = RAG_chain.invoke({"topic": "Prompt Engineering"})
        print(quiz)
        if quiz and "quiz_title" in quiz:
            print(quiz["quiz_title"])

if __name__ == "__main__":
    main()
