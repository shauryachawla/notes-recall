import os

from dotenv import load_dotenv
from data_cleaning.ingestion import Note, fetch_notes
from data_cleaning.rag import RAG
from db.chromadb_client import ChromaDBClient
from datetime import datetime
from llm.llm_client import LLMClient

load_dotenv()

db = ChromaDBClient.initialize()

#TODO: Refactor the codebase
#TODO: add title support
#TODO: add support to ignore labels

def persist_rag_notes(note_data_objects, client) -> int:
    """Persist note_data_objects into ChromaDB 'notes_recall.rag_notes'.
    Returns count of inserted documents.
    """
    if not note_data_objects:
        return 0

    collection = client.get_or_create_collection(name="notes_recall.rag_notes")
    documents = []
    metadatas = []
    ids = []
    embeddings = []

    for i, n in enumerate(note_data_objects):
        if hasattr(n, "model_dump"):
            doc = n.model_dump()
        elif hasattr(n, "dict"):
            doc = n.dict()
        else:
            try:
                doc = dict(n)
            except Exception:
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
        ids.append(f"note-{i}")
        embeddings.append(embedding)

    if not documents:
        return 0

    collection.add(
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
        # if (len(note_data_objects) > 10):
        #     break
        split_notes = []
        for chunk in rag.split_text(note.text):
            chunk = " ".join(chunk.split())

            split_note = Note.model_validate({
                "embedding": rag.get_embedding(chunk),
                "text": chunk,
                "creation_date": datetime.fromtimestamp(note.creation_date / 1000000.0).strftime('%d-%m-%Y %H:%M:%S.%f'),
                "isArchived": note.isArchived,
            })
            split_notes.append(split_note)
        note_data_objects.extend(split_notes)

    print(f"Fetched {len(note_data_objects)} split notes.")
    inserted = persist_rag_notes(note_data_objects, db.client)
    print(f"Inserted {inserted} notes into notes-recall.rag_notes.")


if __name__ == "__main__":
    # fetch_and_persist_notes_and_embeddings()
    topic = input("What topic do you want to ask about?")
    query = input("What's your query?")
    llm = LLMClient()
    query_emb = llm.embed_query(topic)
    collection = db.client.get_collection("notes_recall.rag_notes")
    results = collection.query(query_embeddings=[query_emb], n_results=5)
    print(llm.generate(prompt=query, system=None, context=results.get("documents", [[]])[0]))
