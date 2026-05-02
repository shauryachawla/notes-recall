 

from dotenv import load_dotenv
from data_cleaning.ingestion import Note, fetch_notes
from data_cleaning.rag import RAG
from db.chromadb_client import ChromaDBClient
from datetime import datetime
from llm.llm_client import LLMClient
import hashlib

load_dotenv()

db = ChromaDBClient.initialize()

#TODO: Make this more generic. divide into separate files if needed. The main.py should ideally just be the entry point that calls into other modules for ingestion, querying, etc.

def persist_rag_notes(note_data_objects, client) -> int:
    """Persist note_data_objects into ChromaDB 'notes_recall.rag_notes'.
    Returns count of inserted documents.
    """
    if not note_data_objects:
        return 0

    rag_collection = client.get_or_create_collection(name="notes_recall.rag_notes")
    documents = []
    metadatas = []
    ids = []
    embeddings = []

    for n in note_data_objects:
        if hasattr(n, "model_dump"):
            doc = n.model_dump()
        elif hasattr(n, "dict"):
            doc = n.dict()
        else:
            try:
                doc = dict(n)
            except (TypeError, ValueError):
                doc = {}

        text = doc.get("text")
        embedding = doc.get("embedding")

        if not text or embedding is None:
            continue

        documents.append(text)
        metadatas.append({
            "creation_date": doc.get("creation_date"),
            "isArchived": doc.get("isArchived", False),
        })
        # Deterministic ID to make ingestion idempotent
        ids.append(hashlib.blake2s(text.encode('utf-8')).hexdigest())
        embeddings.append(embedding)

    if not documents:
        return 0

    # Prefer upsert to avoid duplicate ID errors on re-ingestion
    try:
        upsert = getattr(rag_collection, "upsert")
        upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
    except AttributeError:
        rag_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    return len(documents)


def fetch_and_persist_notes_and_embeddings():
    """Fetch notes from the Keep resource and print the count."""
    notes = fetch_notes('resources/Keep')
    rag = RAG(model_name="text-embedding-3-small")
    note_data_objects = []
    for note in notes:
        # Split note text and embed in a single batch call for throughput
        chunks = [" ".join(chunk.split()) for chunk in rag.split_text(note.text)]
        if not chunks:
            continue
        embeddings = rag.get_embeddings(chunks)
        split_notes = [
            Note.model_validate({
                "embedding": emb,
                "title": note.title,
                "labels": note.labels,
                "text": chunk,
                "creation_date": datetime.fromtimestamp(note.creation_date / 1000000.0).strftime('%d-%m-%Y %H:%M:%S.%f'),
                "isArchived": note.isArchived,
            })
            for chunk, emb in zip(chunks, embeddings)
        ]
        note_data_objects.extend(split_notes)

    print(f"Fetched {len(note_data_objects)} split notes.")
    inserted = persist_rag_notes(note_data_objects, db.client)
    print(f"Inserted {inserted} notes into notes-recall.rag_notes.")


if __name__ == "__main__":
    # fetch_and_persist_notes_and_embeddings()
    topic = input("What topic do you want to ask about?")
    query = input("What's your query?")
    llm = LLMClient()
    query_emb = llm.embed_query(query)
    try:
        collection = db.client.get_collection("notes_recall.rag_notes")
    except Exception:
        print("No collection found. Please run ingestion first.")
        raise
    results = collection.query(query_embeddings=[query_emb], n_results=5)
    docs = results.get("documents", [[]])[0]
    CTX = "\n\n".join(docs) if docs else ""
    print(llm.generate(prompt=query, system=None, context=CTX))
