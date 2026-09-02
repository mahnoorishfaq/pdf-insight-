from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.pdf_processor import Chunk

# Small, fast, and good enough for semantic search over a document.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    """A minimal in-memory vector store backed by a FAISS index."""

    def __init__(self, model: SentenceTransformer | None = None, model_name: str = DEFAULT_MODEL_NAME):
        self.model = model or SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """
        Embeds a list of chunks and builds a fresh FAISS index over
        them. Cosine similarity is approximated by normalizing
        embeddings and using an inner-product index.

        Args:
            chunks: The Chunk objects to index.
        """
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Finds the chunks most semantically similar to the query.

        Args:
            query: The user's natural-language question.
            top_k: How many chunks to return.

        Returns:
            A list of (Chunk, similarity_score) tuples, best first.
        """
        if self.index is None or not self.chunks:
            return []

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding, min(top_k, len(self.chunks))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results
