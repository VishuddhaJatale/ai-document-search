from langchain_core.documents import Document

from app import retriever


class FakeStore:
    def __init__(self):
        self.docs = [
            Document(page_content="The attendance policy requires 75 percent attendance.", metadata={"chunk_id": "c1", "document_id": "d1"}),
            Document(page_content="Students must submit examination forms before deadlines.", metadata={"chunk_id": "c2", "document_id": "d1"}),
            Document(page_content="Attendance shortage may require a condonation fee.", metadata={"chunk_id": "c3", "document_id": "d2"}),
        ]

    def similarity_search_with_relevance_scores(self, question, k, filter=None):
        docs = [doc for doc in self.docs if not filter or doc.metadata["document_id"] == filter["document_id"]]
        return [(doc, 0.8 - index * 0.1) for index, doc in enumerate(docs[:k])]

    def max_marginal_relevance_search(self, question, k, fetch_k, filter=None):
        return [doc for doc, _ in self.similarity_search_with_relevance_scores(question, fetch_k, filter)[:k]]

    def get(self, where=None, include=None):
        docs = [doc for doc in self.docs if not where or doc.metadata["document_id"] == where["document_id"]]
        result = {}
        if include and "documents" in include:
            result["documents"] = [doc.page_content for doc in docs]
        if include and "metadatas" in include:
            result["metadatas"] = [doc.metadata for doc in docs]
        return result


def test_bm25_prefers_exact_terms():
    docs = [
        Document(page_content="Students should attend lectures regularly."),
        Document(page_content="Attendance requirement is 75 percent."),
    ]
    ranked = retriever._bm25_rank("attendance 75 percent", docs, 2)
    assert ranked[0].page_content == docs[1].page_content


def test_hybrid_retrieval_combines_dense_and_lexical(monkeypatch):
    monkeypatch.setattr(retriever, "get_vector_store", FakeStore)
    monkeypatch.setattr(retriever, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(retriever, "USE_MMR", False)
    monkeypatch.setattr(retriever, "MIN_RELEVANCE_SCORE", 0.0)
    results = retriever.retrieve("attendance 75 percent", k=2)
    assert len(results) == 2
    assert results[0][0].metadata["chunk_id"] == "c1"
    assert all(score > 0 for _, score in results)


def test_hybrid_respects_document_filter(monkeypatch):
    monkeypatch.setattr(retriever, "get_vector_store", FakeStore)
    monkeypatch.setattr(retriever, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(retriever, "USE_MMR", False)
    monkeypatch.setattr(retriever, "MIN_RELEVANCE_SCORE", 0.0)
    results = retriever.retrieve("attendance", k=5, document_id="d2")
    assert results
    assert all(doc.metadata["document_id"] == "d2" for doc, _ in results)


def test_dense_mmr_threshold_does_not_silently_bypass_filter(monkeypatch):
    monkeypatch.setattr(retriever, "get_vector_store", FakeStore)
    monkeypatch.setattr(retriever, "RETRIEVAL_MODE", "dense")
    monkeypatch.setattr(retriever, "USE_MMR", True)
    monkeypatch.setattr(retriever, "MIN_RELEVANCE_SCORE", 0.5)
    # MMR has no calibrated relevance scores, so the documented behavior is
    # to skip score-threshold filtering rather than pretend the threshold ran.
    results = retriever.retrieve("attendance", k=2)
    assert len(results) == 2
    assert all(score is None for _, score in results)
