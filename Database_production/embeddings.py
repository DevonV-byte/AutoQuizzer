# --- Imports ---
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm
import chromadb

from .document_loader import COURSE_DIR, load_course_documents
from .text_splitter import split_documents

# --- Constants ---
CHROMA_DB_PATH = "../Database/"
COLLECTION_NAME = "autoquizzer_collection"

# --- Functions ---
def get_embeddings_model():
    """
    Initializes and returns the Google Generative AI embeddings model.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)

# --- Main Execution ---
def main():
    """
    Loads documents, splits them into chunks, creates embeddings, and stores them in ChromaDB.
    """
    try:
        # Load and split documents
        print("Loading and splitting documents...")
        documents = load_course_documents(COURSE_DIR)
        chunks = split_documents(documents)
        print(f"Loaded {len(documents)} documents and split them into {len(chunks)} chunks.")

        if not chunks:
            print("No chunks were created from the documents. Exiting.")
            return

        # Initialize embeddings model
        embeddings_model = get_embeddings_model()
        print("Successfully initialized the embeddings model.")

        # Initialize ChromaDB client and collection
        print(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"Using collection: '{COLLECTION_NAME}'")

        # Add documents to ChromaDB in batches
        batch_size = 99  # Stay under the 100 requests/minute free tier limit
        total_chunks = len(chunks)
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="Adding chunks to ChromaDB"):
            batch_chunks = chunks[i:i + batch_size]
            
            batch_texts = [chunk.page_content for chunk in batch_chunks]
            batch_ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
            batch_metadatas = [chunk.metadata for chunk in batch_chunks]

            # Create embeddings for the batch
            batch_embeddings = embeddings_model.embed_documents(batch_texts)

            # Add to ChromaDB
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            # Sleep to respect the rate limit
            if i + batch_size < total_chunks:
                time.sleep(61)  # Sleep for just over a minute

        print("\nSuccessfully added all chunks to ChromaDB.")
        print(f"Total items in collection: {collection.count()}")

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
