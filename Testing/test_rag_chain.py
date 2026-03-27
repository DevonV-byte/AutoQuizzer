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
    assert "difficulty" in prompt.input_variables
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

        # 4. Invoke the chain with the required topic and difficulty keys
        query = {"topic": "What is Prompt Engineering?", "difficulty": "medium"}
        result = chain.invoke(query)

        # 5. Assertions
        mock_vectorstore.as_retriever.assert_called_once()

        # LCEL coerces a plain callable into a RunnableLambda and calls it via __call__,
        # not .invoke(), so we assert on the mock itself rather than mock.invoke.
        mock_llm.assert_called_once()
        llm_input_prompt = mock_llm.call_args[0][0]
        # The topic and difficulty must appear in the formatted prompt
        assert "What is Prompt Engineering?" in llm_input_prompt.to_string()
        assert "medium" in llm_input_prompt.to_string()

        # The final result should be what the parser returns
        assert result == "This is the final parsed answer."
