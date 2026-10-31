from typing import List, Optional, Tuple
from langchain_core.documents import Document
from .config import MIN_RELEVANCE_SCORE, USE_MMR
from .vector_store import get_vector_store


def retrieve(question: str, k: int = 6, document_id: Optional[str] = None) -> List[Tuple[Document, Optional[float]]]:
    store = get_vector_store()
    filters = {"document_id": document_id} if document_id else None
    if USE_MMR:
        docs = store.max_marginal_relevance_search(question, k=k, fetch_k=max(k * 3, 12), filter=filters)
        results = [(doc, None) for doc in docs]
    else:
        results = store.similarity_search_with_relevance_scores(question, k=k, filter=filters)
    if MIN_RELEVANCE_SCORE <= 0:
        return results
    return [(doc, score) for doc, score in results if score is None or score >= MIN_RELEVANCE_SCORE]
