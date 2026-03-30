# --- File Description ---
# Unit tests for the RAG pipeline defined in RAG_Pipeline/rag_chain.py.
# This file uses pytest and mocking to test the functionality of the RAG chain
# components without making actual API calls or accessing the file system.
#
# Created: 2026-03-20
# Author: Devon Vanaenrode
# --- Imports ---

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add project root to path to allow imports from other directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RAG_Pipeline.rag_chain import (
    get_llm_model,
    create_prompt_template,
    format_docs,
    rag_chain,
    invoke_with_fallback,
    DIFFICULTY_TIERS,
    FALLBACK_PROMPT,
)

# --- Globals ---

# Mock Document class for testing
class MockDocument:
    def __init__(self, page_content):
        self.page_content = page_content

# --- Fixtures ---

@pytest.fixture
def mock_llm():
    """Fixture for a mocked LLM model."""
    return MagicMock()

@pytest.fixture
def mock_vectorstore():
    """Fixture for a mocked vector store."""
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [
        MockDocument(page_content="Prompt engineering is the process of designing effective prompts.")
    ]
    
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs

@pytest.fixture
def prompt_template():
    """Fixture to get the prompt template."""
    return create_prompt_template()

# --- Tests ---

def test_get_llm_model_success():
    """
    Tests that get_llm_model successfully returns a ChatGoogleGenerativeAI instance
    when the API key is present.
    """
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test_api_key"}):
        with patch('RAG_Pipeline.rag_chain.ChatGoogleGenerativeAI') as mock_chat_google:
            llm = get_llm_model()
            mock_chat_google.assert_called_once_with(model="models/gemini-2.5-flash-lite", google_api_key="test_api_key")
            assert llm is not None

def test_get_llm_model_no_api_key():
    """
    Tests that get_llm_model raises a ValueError when the GOOGLE_API_KEY is not set.
    """
    with patch.dict(os.environ, {}, clear=True):
        with patch('RAG_Pipeline.rag_chain.load_dotenv'):  # prevent .env file from re-populating the env
            with pytest.raises(ValueError, match="GOOGLE_API_KEY not found in environment variables."):
                get_llm_model()

def test_create_prompt_template():
    """
    Tests that create_prompt_template returns a PromptTemplate with the correct
    input variables and a non-empty template string.
    """
    from langchain_core.prompts import PromptTemplate

    prompt = create_prompt_template()
    assert isinstance(prompt, PromptTemplate)
    assert "context" in prompt.input_variables
    assert "topic" in prompt.input_variables
    # difficulty is resolved at template-build time and injected as partial variables
    assert "tier_label" in prompt.partial_variables
    assert "tier_instructions" in prompt.partial_variables
    assert len(prompt.template) > 0

def test_format_docs():
    """
    Tests that format_docs correctly joins document page contents.
    """
    docs = [
        MockDocument(page_content="This is the first document."),
        MockDocument(page_content="This is the second document."),
    ]
    formatted_string = format_docs(docs)
    expected_string = "This is the first document.\n\nThis is the second document."
    assert formatted_string == expected_string

def test_rag_chain_construction(mock_llm, prompt_template, mock_vectorstore):
    """
    Tests that the rag_chain function constructs a runnable sequence (chain)
    with the correct components.
    """
    from langchain_core.runnables import RunnableSequence

    chain = rag_chain(mock_llm, prompt_template, mock_vectorstore)
    
    # Check if the returned object is a RunnableSequence
    assert isinstance(chain, RunnableSequence)
    
    # You can perform more detailed checks on the chain's structure if needed
    # For example, checking the steps in the chain
    assert len(chain.steps) == 4 # {"context": ..., "topic": ..., "difficulty": ...} | prompt | llm | parser

def test_rag_chain_invocation():
    """
    Tests the end-to-end invocation of the RAG chain with mocked components.
    This test ensures the data flows correctly through the chain.
    """
    # 1. Setup Mocks
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "This is the LLM's answer."

    mock_parser = MagicMock()
    # LCEL coerces MagicMock to RunnableLambda and calls via __call__, not .invoke()
    mock_parser.return_value = "This is the final parsed answer."

    mock_docs = [MockDocument(page_content="Content about prompt engineering.")]
    mock_retriever = MagicMock()
    # The retriever is called via the passed-in context lambda, so we mock its direct output
    # `retriever | format_docs` is what gets called.
    # The retriever part of that will be called with the query.
    mock_retriever.invoke.return_value = mock_docs

    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    prompt = create_prompt_template()

    # 2. Patch JsonOutputParser so we can intercept the output without real JSON parsing
    with patch('RAG_Pipeline.rag_chain.JsonOutputParser', return_value=mock_parser):

        # 3. Create the chain with mocks
        chain = rag_chain(mock_llm, prompt, mock_vectorstore)

        # 4. Invoke the chain with the required topic key
        query = {"topic": "What is Prompt Engineering?"}
        result = chain.invoke(query)

        # 5. Assertions
        mock_vectorstore.as_retriever.assert_called_once()

        # LCEL coerces a plain callable into a RunnableLambda and calls it via __call__,
        # not .invoke(), so we assert on the mock itself rather than mock.invoke.
        mock_llm.assert_called_once()
        llm_input_prompt = mock_llm.call_args[0][0]
        # The topic must appear in the formatted prompt; difficulty is now a partial variable
        # rendered as the tier label (e.g. "Skeleton-level"), not the raw string "medium"
        assert "What is Prompt Engineering?" in llm_input_prompt.to_string()
        assert "Skeleton" in llm_input_prompt.to_string()

        # The final result should be what the parser returns
        assert result == "This is the final parsed answer."

def test_prompt_enforces_strict_json():
    """Prompt must instruct Gemini to return ONLY JSON with no markdown or extra text."""
    prompt = create_prompt_template()
    template_text = prompt.template.lower()
    assert "return only" in template_text
    assert "no markdown" in template_text

def test_prompt_json_structure_uses_braces():
    """JSON structure illustration must use curly braces for objects, not square brackets."""
    prompt = create_prompt_template()
    assert '"quiz_title"' in prompt.template

def test_prompt_contains_required_fields():
    """Prompt template must reference all required output fields."""
    prompt = create_prompt_template()
    for field in ["question", "options", "answer", "explanation"]:
        assert field in prompt.template, f"Required field '{field}' missing from prompt template"

# --- Difficulty Tier Tests ---

def test_slime_tier_prompt_contains_recall_instruction():
    """Easy (Slime-level) prompt must reference recall and recognition."""
    prompt = create_prompt_template(difficulty="easy")
    text = prompt.partial_variables["tier_instructions"].lower()
    assert "recall" in text
    assert "recogni" in text  # covers 'recognition' and 'recognise'

def test_skeleton_tier_prompt_contains_understanding_instruction():
    """Medium (Skeleton-level) prompt must reference understanding and application."""
    prompt = create_prompt_template(difficulty="medium")
    text = prompt.partial_variables["tier_instructions"].lower()
    assert "understanding" in text
    assert "appli" in text  # covers 'application' and 'apply'

def test_dragon_tier_prompt_contains_synthesis_instruction():
    """Hard (Dragon-level) prompt must reference synthesis and analysis."""
    prompt = create_prompt_template(difficulty="hard")
    text = prompt.partial_variables["tier_instructions"].lower()
    assert "synthesis" in text or "synthesize" in text or "integrat" in text
    assert "analys" in text  # covers 'analysis' and 'analyse'

def test_all_tiers_produce_distinct_prompts():
    """Each difficulty tier must produce different tier_instructions text."""
    instructions = [
        create_prompt_template(difficulty=d).partial_variables["tier_instructions"]
        for d in ["easy", "medium", "hard"]
    ]
    assert len(set(instructions)) == 3, "All three tiers must have unique instructions"

def test_dragon_tier_retriever_uses_more_chunks():
    """Hard tier must have a higher retriever k than easy tier."""
    assert DIFFICULTY_TIERS["hard"][2] > DIFFICULTY_TIERS["easy"][2]

def test_invalid_difficulty_raises_value_error():
    """Unsupported difficulty string must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid difficulty"):
        create_prompt_template(difficulty="legendary")

# --- JSON Output Parsing Tests ---

# Realistic fixture strings matching the prompt's output schema, one per tier.
TIER_FIXTURES = {
    "easy": """{
        "quiz_title": "Python Basics Quiz",
        "difficulty": "Tier 1 - Slime-level",
        "questions": [
            {
                "question_number": 1,
                "question": "What keyword is used to define a function in Python?",
                "options": {"A": "func", "B": "def", "C": "define"},
                "answer": "B",
                "explanation": "The 'def' keyword is used to define a function in Python."
            }
        ]
    }""",
    "medium": """{
        "quiz_title": "RAG Architecture Quiz",
        "difficulty": "Tier 2 - Skeleton-level",
        "questions": [
            {
                "question_number": 1,
                "question": "How does retrieval-augmented generation improve LLM responses?",
                "options": {"A": "By fine-tuning the model", "B": "By injecting retrieved context into the prompt", "C": "By increasing model size"},
                "answer": "B",
                "explanation": "RAG injects relevant retrieved documents into the prompt so the model can ground its response in external knowledge."
            }
        ]
    }""",
    "hard": """{
        "quiz_title": "LLM Evaluation and Agents Quiz",
        "difficulty": "Tier 3 - Dragon-level",
        "questions": [
            {
                "question_number": 1,
                "question": "How do evaluation metrics for RAG systems differ from those used for standalone LLM generation, and what trade-offs arise when using agents?",
                "options": {"A": "They are identical; agents add no trade-offs", "B": "RAG metrics assess retrieval quality separately; agents introduce latency and error propagation", "C": "RAG metrics only measure fluency; agents eliminate hallucination"},
                "answer": "B",
                "explanation": "RAG evaluation must assess both retrieval relevance and generation faithfulness. Agents add orchestration complexity, increasing latency and the risk of compounding errors across tool calls."
            }
        ]
    }""",
}

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_json_parser_parses_to_dict(difficulty):
    """JsonOutputParser must return a Python dict for each tier's fixture string."""
    from langchain_core.output_parsers import JsonOutputParser
    result = JsonOutputParser().parse(TIER_FIXTURES[difficulty])
    assert isinstance(result, dict)

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_json_parser_top_level_keys(difficulty):
    """Parsed output must contain quiz_title, difficulty, and questions."""
    from langchain_core.output_parsers import JsonOutputParser
    result = JsonOutputParser().parse(TIER_FIXTURES[difficulty])
    for key in ["quiz_title", "difficulty", "questions"]:
        assert key in result, f"Missing top-level key '{key}' for tier '{difficulty}'"

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_json_parser_questions_have_required_fields(difficulty):
    """Every question in parsed output must have all required fields."""
    from langchain_core.output_parsers import JsonOutputParser
    result = JsonOutputParser().parse(TIER_FIXTURES[difficulty])
    required = ["question_number", "question", "options", "answer", "explanation"]
    for question in result["questions"]:
        for field in required:
            assert field in question, f"Missing field '{field}' in question for tier '{difficulty}'"

@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_json_parser_options_is_dict(difficulty):
    """The options field in each question must be a dict."""
    from langchain_core.output_parsers import JsonOutputParser
    result = JsonOutputParser().parse(TIER_FIXTURES[difficulty])
    for question in result["questions"]:
        assert isinstance(question["options"], dict), f"options is not a dict for tier '{difficulty}'"

# --- Fallback Tests ---

VALID_QUIZ_JSON = TIER_FIXTURES["medium"]

def test_fallback_not_triggered_on_success():
    """LLM must be called exactly once when the chain succeeds on the first attempt."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"quiz_title": "Test", "difficulty": "medium", "questions": []}
    mock_llm = MagicMock()

    invoke_with_fallback(mock_chain, mock_llm, "Prompt Engineering")

    mock_chain.invoke.assert_called_once_with({"topic": "Prompt Engineering"})
    mock_llm.invoke.assert_not_called()

def test_fallback_retries_on_parse_failure():
    """LLM must be called a second time when the chain raises OutputParserException."""
    from langchain_core.exceptions import OutputParserException

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = OutputParserException("bad json")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=VALID_QUIZ_JSON)

    invoke_with_fallback(mock_chain, mock_llm, "RAG")

    mock_chain.invoke.assert_called_once()
    mock_llm.invoke.assert_called_once()

def test_fallback_returns_dict_after_retry():
    """Result after a successful retry must be a dict with expected keys."""
    from langchain_core.exceptions import OutputParserException

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = OutputParserException("bad json")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=VALID_QUIZ_JSON)

    result = invoke_with_fallback(mock_chain, mock_llm, "RAG")

    assert isinstance(result, dict)
    assert "quiz_title" in result
    assert "questions" in result

def test_fallback_raises_after_two_failures():
    """An exception must propagate if the retry response also fails to parse."""
    from langchain_core.exceptions import OutputParserException

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = OutputParserException("bad json")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="this is not json at all")

    with pytest.raises(Exception):
        invoke_with_fallback(mock_chain, mock_llm, "RAG")

def test_fallback_retry_prompt_is_stricter():
    """The fallback prompt template must contain a stricter JSON-only instruction."""
    prompt_text = FALLBACK_PROMPT.lower()
    assert "no markdown" in prompt_text
    assert "no code fences" in prompt_text
    assert "json" in prompt_text
