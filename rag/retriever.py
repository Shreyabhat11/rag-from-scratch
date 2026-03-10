"""retriever.py — High-level retriever that ties Embedder + VectorStore."""
from typing import List, Tuple
from rag.embedder import Embedder
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Embed query and return top_k (chunk, score) pairs."""
        q_vec = self.embedder.encode_query(query)
        return self.store.search(q_vec, top_k=top_k)

    def get_context(self, query: str, top_k: int = 3) -> str:
        """Return retrieved chunks joined as a single context string for an LLM."""
        results = self.retrieve(query, top_k=top_k)
        return "\n\n---\n\n".join(chunk for chunk, _ in results)
