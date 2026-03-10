import numpy as np
from typing import List, Tuple

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False
    print("[VectorStore] FAISS not found — using NumPy brute-force.")


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: List[str] = []
        if _FAISS:
            # Inner Product on L2-normalised vecs == cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.matrix: np.ndarray = np.empty((0, dim), dtype=np.float32)

    def add(self, embeddings: np.ndarray, chunks: List[str]):
        """Add embeddings + their source chunks to the store."""
        self.chunks.extend(chunks)
        if _FAISS:
            self.index.add(embeddings)
        else:
            self.matrix = np.vstack([self.matrix, embeddings])
        print(f"[VectorStore] Indexed {len(chunks)} chunks. Total: {len(self.chunks)}")

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return top_k (chunk, score) pairs for a query vector."""
        if _FAISS:
            scores, indices = self.index.search(query_vec, top_k)
            results = [(self.chunks[i], scores[0][j])
                       for j, i in enumerate(indices[0]) if i >= 0]
        else:
            sims = self.matrix @ query_vec.T          # cosine similarity
            top_idx = np.argsort(sims.ravel())[::-1][:top_k]
            results = [(self.chunks[i], float(sims[i])) for i in top_idx]
        return results