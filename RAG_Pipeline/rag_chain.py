# Create the core RAG chain using LCEL
# 
# Created: 2026-03-20
# Author: Devon Vanaenrode
# Updated: 2026-03-25
# --- Imports ---
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
import json

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

from Database_production.document_loader import COURSE_DIR, load_course_documents
from Database_production.text_splitter import split_documents
from Database_production import embeddings

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(SCRIPT_DIR, "..", "Database")
COLLECTION_NAME = "autoquizzer_collection"

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

def create_prompt_template(n_questions=5, n_options=3):
    prompt_template = """
    Use the information from the documents to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

    {context}

    Question: {question}
    """
    
    quiz_template = """
    You are a quiz master. The user specifies which topic they want to be quizzed on. Use the information from the documents to construct a quiz:
    Number of question: {n_questions}
    Number of options per question: {n_options}
    JSON structure (x signals placeholders):
        [ 'quiz_title': x, 'questions':
            [
                'question_number': x,
                'question': x,
                'options': ['A' : x, 'B' : x, ... (based on number of questions)],
                'answer': x,
                'explanation': x
            ],
                'question_number': x,
                    'question': x,
                    'options': ['A' : x, 'B' : x, ... (based on number of options)],
                    'answer': x,
                    'explanation': x
            ],
            ... (based on number of questions)
        ]
    {context}

    Topic: {topic}
    """

    prompt_template = PromptTemplate(
        template=quiz_template, 
        input_variables=["context", "topic"],
        partial_variables={"n_questions": n_questions, "n_options": n_options}
    )

    return prompt_template

def rag_chain(llm_model, prompt_template, vectorstore):
    retriever = vectorstore.as_retriever()
    
    # Creates a LangChain Runnable using LCEL
    qa_chain = (
        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
        | prompt_template
        | llm_model
        | JsonOutputParser()
    )
    
    return qa_chain

def get_quiz_generation_chain():
    """
    Initializes and returns the quiz generation RAG chain.
    """
    try:
        llm_model = get_llm_model()
        prompt = create_prompt_template(3, 3) # Using default values for now
        embeddings_model = embeddings.get_embeddings_model()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings_model
        )
        return rag_chain(llm_model, prompt, vectorstore)
    except (ValueError, FileNotFoundError) as e:
        # Should probably log this or handle it better
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
        query = "Prompt Engineering"
        quiz = RAG_chain.invoke(query)
        print(quiz)
        if quiz and "quiz_title" in quiz:
            print(quiz["quiz_title"])

if __name__ == "__main__":
    main()
