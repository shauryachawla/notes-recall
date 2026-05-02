import chromadb
from dotenv import load_dotenv
try:
    from data_cleaning.rag import RAG
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data_cleaning.rag import RAG
    from llm.llm_client import LLMClient


load_dotenv()

class ChromaDBClient:
    """A simple ChromaDB client wrapper."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")

    @classmethod
    def initialize(cls):
        """Initialize the ChromaDB client."""
        return cls()
    
    def getAll(self):
        print(self.client.get_collection("notes_recall.rag_notes").get())

