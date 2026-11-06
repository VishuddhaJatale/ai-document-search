import math
import re
from collections import Counter
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from .config import MIN_RELEVANCE_SCORE, RETRIEVAL_MODE, USE_MMR
from .vector_store import get_vector_store

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_RRF_K = 60


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_rank(question: str, documents: list[Document], limit: int) -> list[Document]:
    """Small dependency-free BM25 implementation for the current local corpus."""
    query_terms = list(dict.fromkeys(_tokens(question)))
    if not query_terms or not documents:
        return []
    tokenized = [_tokens(doc.page_content) for doc in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency = Counter(term for terms in tokenized for term in set(terms))
    scores = []
    for index, terms in enumerate(tokenized):
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            idf = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * lengths[index] / max(average_length, 1))
            score += idf * frequency * 2.5 / denominator
        scores.append((score, index))
    scores.sort(key=lambda item: item[0], reverse=True)
    return [documents[index] for score, index in scores[:limit] if score > 0]


def _dense_results(store, question: str, k: int, filters):
    if USE_MMR:
        docs = store.max_marginal_relevance_search(question, k=k, fetch_k=max(k * 3, 12), filter=filters)
        return [(doc, None) for doc in docs]
    results = store.similarity_search_with_relevance_scores(question, k=k, filter=filters)
    if MIN_RELEVANCE_SCORE > 0:
        results = [(doc, score) for doc, score in results if score >= MIN_RELEVANCE_SCORE]
    return results


def retrieve(question: str, k: int = 6, document_id: Optional[str] = None) -> List[Tuple[Document, Optional[float]]]:
    store = get_vector_store()
    filters = {"document_id": document_id} if document_id else None
    dense = _dense_results(store, question, max(k * 3, 12), filters)
    if RETRIEVAL_MODE == "dense" or not dense:
        return dense[:k]

    # Build lexical candidates from the same filtered Chroma corpus. This avoids
    # maintaining a second index that could become stale after document deletion.
    where = filters
    corpus = store.get(where=where, include=["documents", "metadatas"])
    texts = corpus.get("documents") or []
    metadatas = corpus.get("metadatas") or []
    lexical_docs = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(texts, metadatas)
        if content
    ]
    lexical = _bm25_rank(question, lexical_docs, max(k * 3, 12))
    if not lexical:
        return dense[:k]

    fused = {}
    documents = {}
    for rank, (doc, _score) in enumerate(dense, 1):
        key = doc.metadata.get("chunk_id") or (doc.metadata.get("document_id"), doc.metadata.get("page"), doc.page_content)
        documents[key] = doc
        fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, doc in enumerate(lexical, 1):
        key = doc.metadata.get("chunk_id") or (doc.metadata.get("document_id"), doc.metadata.get("page"), doc.page_content)
        documents[key] = doc
        fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(documents[key], score) for key, score in ranked[:k]]
