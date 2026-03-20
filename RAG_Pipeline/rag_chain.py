# Create the core RAG chain using LCEL
# 
# Created: 2026-03-20
# Author: Devon Vanaenrode
# --- Imports ---
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from Database_production.document_loader import COURSE_DIR, load_course_documents
from Database_production.text_splitter import split_documents
from Database_production import embeddings

# --- Constants ---
CHROMA_DB_PATH = "../Database/"
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
    
    return ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", google_api_key=api_key)

def create_prompt_template():
    prompt_template = """
    Use the information from the documents to answer the question at the end. If you don't know the answer, just say that you don't know, definately do not try to make up an answer.

    {context}

    Question: {question}
    """

    prompt_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return prompt_template

def rag_chain(llm_model, prompt_template, vectorstore):
    retriever = vectorstore.as_retriever()
    
    # This creates a true LangChain Runnable using LCEL
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm_model
        | StrOutputParser()
    )
    
    return qa_chain

# --- Main Execution ---
def main():
    """
    Create the LLM model, connect ChromaDB to the prompt, pipe it into Gemini and test with simple string.
    """
    try:
        # Initialize LLM model
        llm_model = get_llm_model()
        # print("Successfully initialized the LLM model.")

        # Create the prompt
        chain_type_kwargs = create_prompt_template()
        # print("Successfully created the prompt.")

        # Get documents from Chromadb
        embeddings_model = embeddings.get_embeddings_model()
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings_model
        )
        RAG_chain = rag_chain(llm_model, chain_type_kwargs, vectorstore)

        # Test our retrievalQA
        query = "What is Prompt Engineering?"
        print(RAG_chain.invoke(query))

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    

if __name__ == "__main__":
    main()
