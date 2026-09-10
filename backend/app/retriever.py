from typing import List, Optional, Tuple

from langchain_core.documents import Document

from .config import MIN_RELEVANCE_SCORE
from .vector_store import get_vector_store


def retrieve(
    question: str,
    k: int = 6,
    document_id: Optional[str] = None,
) -> List[Tuple[Document, Optional[float]]]:
    """Retrieve the most relevant chunks, optionally restricted to one document."""
    store = get_vector_store()
    filters = {"document_id": document_id} if document_id else None

    results = store.similarity_search_with_relevance_scores(
        question,
        k=k,
        filter=filters,
    )

    # Keep filtering configurable. With the current Chroma setup, a score can
    # legitimately be negative, so the default threshold is -1.0 (no filtering).
    if MIN_RELEVANCE_SCORE <= -1.0:
        return results

    return [
        (doc, score)
        for doc, score in results
        if score is None or score >= MIN_RELEVANCE_SCORE
    ]
