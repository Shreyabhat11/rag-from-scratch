# from typing import List


# def chunk_text(
#     text: str,
#     chunk_size: int = 10,
#     overlap: int = 2,
# ) -> List[str]:
#     """Split text into overlapping character-level chunks."""
#     chunks, start = [], 0
#     words = text.split()
#     while start < len(words):
#         end = start + chunk_size
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start += chunk_size - overlap   # slide with overlap
#     return chunks


# def chunk_documents(docs: List[str], **kwargs) -> List[str]:
#     """Chunk a list of documents."""
#     all_chunks = []
#     for doc in docs:
#         all_chunks.extend(chunk_text(doc, **kwargs))
#     return all_chunks

"""
chunker.py — Three chunking strategies for RAG.

Strategies (in order of quality for RAG):
  1. paragraph   — split on blank lines (best: preserves natural topic boundaries)
  2. sentence    — split on sentence endings (good: fine-grained, precise)
  3. fixed_word  — sliding window over words (baseline: ignores structure)

Rule of thumb:
  - Use 'paragraph' first — if your docs have clear paragraphs, this wins.
  - Use 'sentence' for dense docs where paragraphs are long.
  - Use 'fixed_word' only when text has no structure (e.g. OCR output).
"""
import re
from typing import List, Literal

Strategy = Literal["paragraph", "sentence", "fixed_word"]


# ── 1. Paragraph chunking ────────────────────────────────────────────────────

def _chunk_by_paragraph(text: str, min_words: int = 10) -> List[str]:
    """
    Split on one or more blank lines.
    Filters out very short paragraphs (headings, stray newlines).
    """
    raw = re.split(r"\n{2,}", text.strip())
    chunks = []
    for para in raw:
        para = para.strip()
        if len(para.split()) >= min_words:
            chunks.append(para)
    return chunks


# ── 2. Sentence chunking ─────────────────────────────────────────────────────

def _chunk_by_sentence(
    text: str,
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split into sentences, then group N sentences per chunk with overlap.

    overlap_sentences=1 means the last sentence of chunk[i]
    is the first sentence of chunk[i+1] — preserving context at boundaries.
    """
    # Sentence splitter: split after . ! ? followed by whitespace + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks, i = [], 0
    while i < len(sentences):
        group = sentences[i : i + sentences_per_chunk]
        chunks.append(" ".join(group))
        i += sentences_per_chunk - overlap_sentences
    return chunks


# ── 3. Fixed-word sliding window (original) ──────────────────────────────────

def _chunk_fixed_word(
    text: str,
    chunk_size: int = 40,      # ← smaller default (was 100)
    overlap: int = 10,
) -> List[str]:
    """Sliding window over words. chunk_size=40 ≈ 2–3 sentences."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: Strategy = "paragraph",
    # paragraph kwargs
    min_words: int = 10,
    # sentence kwargs
    sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
    # fixed_word kwargs
    chunk_size: int = 40,
    overlap: int = 10,
) -> List[str]:
    """
    Split a single document string into chunks.

    Args:
        text:               Raw document text.
        strategy:           'paragraph' | 'sentence' | 'fixed_word'
        min_words:          (paragraph) Skip paragraphs shorter than this.
        sentences_per_chunk:(sentence) Sentences grouped per chunk.
        overlap_sentences:  (sentence) Sentences shared between adjacent chunks.
        chunk_size:         (fixed_word) Words per chunk.
        overlap:            (fixed_word) Words of overlap.

    Returns:
        List of non-empty chunk strings.
    """
    if strategy == "paragraph":
        return _chunk_by_paragraph(text, min_words=min_words)
    elif strategy == "sentence":
        return _chunk_by_sentence(
            text,
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=overlap_sentences,
        )
    elif strategy == "fixed_word":
        return _chunk_fixed_word(text, chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: paragraph | sentence | fixed_word")


def chunk_documents(docs: List[str], **kwargs) -> List[str]:
    """Chunk a list of documents. All kwargs forwarded to chunk_text."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc, **kwargs))
    return all_chunks
