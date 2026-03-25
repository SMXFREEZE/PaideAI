"""
Vector Store
============
ChromaDB-backed semantic search over student notes.
Stores note chunks as embeddings; retrieves the most relevant
passages for any question to ground Claude's responses.

Design:
  - One persistent ChromaDB collection: "student_notes"
  - Each document: {text, label, page, source_file, ingested_at}
  - On query: return top-k chunks with metadata
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


DB_DIR = Path(__file__).parents[2] / "data" / "chroma_db"
COLLECTION_NAME = "student_notes"
DEFAULT_TOP_K = 4


class NoteVectorStore:
    def __init__(self, db_dir: Path = DB_DIR):
        self.db_dir = db_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict], source_file: str) -> int:
        """
        Add note chunks to the vector store.
        chunks: [{page, label, text}, ...]
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        documents, metadatas, ids = [], [], []
        now = datetime.now().isoformat()

        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                continue

            # Stable ID based on content hash
            doc_id = hashlib.sha256(
                f"{source_file}:{chunk['page']}:{text[:100]}".encode()
            ).hexdigest()[:16]

            documents.append(text)
            metadatas.append({
                "label": chunk.get("label", source_file),
                "page": str(chunk.get("page", 1)),
                "source_file": source_file,
                "ingested_at": now,
            })
            ids.append(doc_id)

        if documents:
            # ChromaDB upsert handles duplicates gracefully
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        return len(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        label_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search over student notes.
        Returns list of {text, label, page, source_file, distance}
        """
        count = self.collection.count()
        if count == 0:
            return []

        where = {"label": label_filter} if label_filter else None
        actual_k = min(top_k, count)

        results = self.collection.query(
            query_texts=[question],
            n_results=actual_k,
            where=where,
        )

        chunks = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else 0.0
            chunks.append({
                "text": doc,
                "label": meta.get("label", ""),
                "page": meta.get("page", "1"),
                "source_file": meta.get("source_file", ""),
                "distance": round(dist, 4),
            })

        return chunks

    def format_for_prompt(self, chunks: list[dict]) -> str:
        """Format retrieved notes as a clean block for Claude context."""
        if not chunks:
            return "(No relevant notes found. Teach from first principles.)"

        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Note {i} from '{c['label']}', page {c['page']}]\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_note_labels(self) -> list[str]:
        """Return all unique note labels (uploaded file names)."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        labels = {m["label"] for m in results["metadatas"]}
        return sorted(labels)

    def count(self) -> int:
        return self.collection.count()

    def delete_by_label(self, label: str) -> None:
        self.collection.delete(where={"label": label})
