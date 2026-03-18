# --- Imports ---
import os
import chromadb
import pytest

from Database_production.embeddings import CHROMA_DB_PATH, COLLECTION_NAME

# --- Tests ---
def test_database_connection_and_population():
    """
    Tests if the ChromaDB database can be connected to and if it contains any data.
    """
    # The ChromaDB path is relative to the 'Code' directory where the script is run
    db_path = os.path.join(os.path.dirname(__file__), '..', CHROMA_DB_PATH)

    assert os.path.exists(db_path), f"ChromaDB path '{db_path}' does not exist. Run the embeddings script from the 'Code' directory first."

    client = chromadb.PersistentClient(path=db_path)
    
    # Test if the collection exists
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except ValueError:
        pytest.fail(f"Collection '{COLLECTION_NAME}' not found in the database. Run the embeddings script first.")

    # Test if the collection is populated
    count = collection.count()
    print(f"Found {count} items in the collection '{COLLECTION_NAME}'.")
    assert count > 0, "ChromaDB collection is empty."

    # Test the structure of a single item
    item = collection.get(limit=1, include=["metadatas", "documents", "embeddings"])
    assert item, "Failed to retrieve an item from the collection."
    
    assert all(key in item and len(item[key]) == 1 for key in ["ids", "documents", "metadatas", "embeddings"])
    
    print("Successfully verified database connection, population, and item structure.")
