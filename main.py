from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever

QUERIES = [
    "What is the attention mechanism in transformers?",
    "How does gradient descent optimization work?",
    "What is RAG and why does it reduce hallucinations?",
]

# ── Shared: load docs + embedder (reused across all strategies) ──────────────
docs = load_documents("data")
embedder = Embedder()


def run_strategy(strategy: str, label: str, chunk_kwargs: dict):
    print(f"\n{'='*65}")
    print(f"  STRATEGY: {label}")
    print(f"{'='*65}")

    chunks = chunk_documents(docs, strategy=strategy, **chunk_kwargs)
    print(f"  Chunks created: {len(chunks)}")

    embeddings = embedder.encode(chunks, show_progress=False)
    store = VectorStore(dim=embedder.dim)
    store.add(embeddings, chunks)

    retriever = Retriever(store, embedder)

    for query in QUERIES:
        print(f"\n  🔍 \"{query}\"")
        print(f"  {'─'*60}")
        for rank, (chunk, score) in enumerate(retriever.retrieve(query, top_k=2), 1):
            preview = chunk[:110].replace("\n", " ")
            print(f"  [{rank}] {score:.4f} → {preview}...")


# ── Strategy 1: paragraph (recommended for structured docs) ──────────────────
run_strategy(
    strategy="paragraph",
    label="PARAGRAPH  (split on blank lines — best for structured docs)",
    chunk_kwargs={"min_words": 10},
)

# ── Strategy 2: sentence (good for dense prose) ──────────────────────────────
run_strategy(
    strategy="sentence",
    label="SENTENCE   (3 sentences/chunk, 1 sentence overlap)",
    chunk_kwargs={"sentences_per_chunk": 3, "overlap_sentences": 1},
)

# ── Strategy 3: fixed_word — smaller window than before (baseline) ───────────
run_strategy(
    strategy="fixed_word",
    label="FIXED WORD (40-word window, 10-word overlap — simple baseline)",
    chunk_kwargs={"chunk_size": 40, "overlap": 10},
)