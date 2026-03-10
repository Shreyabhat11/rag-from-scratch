import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Embedding dim: {self.dim}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings → float32 numpy array
        Shape: (len(texts), self.dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True, 
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → shape (1, dim)."""
        return self.encode([query], show_progress=False)