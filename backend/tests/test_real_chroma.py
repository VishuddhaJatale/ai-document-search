"""Integration test using a real (temporary) Chroma store with fake embeddings.

The other tests mock the vector store, so they cannot catch bugs inside
vector_store.py itself. This one exercises the real code path.
"""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import document_service, vector_store


@pytest.fixture
def real_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_DIR", tmp_path / "chroma")
    monkeypatch.setattr(vector_store, "_vector_store", None)
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(document_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(document_service, "REGISTRY_PATH", tmp_path / "registry.json")
    # Undo the autouse mocks from conftest so the real functions are used.
    monkeypatch.setattr(document_service, "find_document_by_hash", vector_store.find_document_by_hash)
    monkeypatch.setattr(document_service, "add_chunks", vector_store.add_chunks)
    monkeypatch.setattr(document_service, "delete_document", vector_store.delete_document)


def test_duplicate_upload_and_delete_with_real_chroma(real_store):
    content = b"Attendance must be at least 75 percent. " * 40
    first = document_service.index_document("a.txt", content)
    second = document_service.index_document("a.txt", content)
    assert first.status == "indexed"
    assert second.status == "already_exists"
    assert second.document_id == first.document_id
    assert document_service.remove_document(first.document_id) is True
    assert document_service.remove_document(first.document_id) is False


def test_concurrent_identical_uploads_create_one_registry_entry(real_store):
    """Regression: an in-flight upload's chunks must not be adopted as a legacy record."""
    import json
    import threading

    for trial in range(10):
        content = f"Trial {trial}. Attendance must be at least 75 percent. ".encode() * 60
        statuses = []
        threads = [threading.Thread(target=lambda: statuses.append(document_service.index_document("c.txt", content).status)) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert statuses.count("indexed") == 1
    registry = json.loads(document_service.REGISTRY_PATH.read_text())
    ids = [item["document_id"] for item in registry]
    assert len(ids) == len(set(ids)) == 10


def test_inflight_hash_is_not_adopted_as_legacy_record(real_store):
    content = b"Some document that is currently being indexed. " * 30
    first = document_service.index_document("a.txt", content)
    document_service.REGISTRY_PATH.write_text("[]")  # chunks exist in Chroma, registry record not written yet
    file_hash = document_service._file_hash(content)
    document_service._inflight[file_hash] = __import__("threading").Event()
    try:
        with document_service._registry_lock:
            assert document_service._find_duplicate_locked("a.txt", file_hash) is None
        assert document_service._read_registry() == []
    finally:
        document_service._inflight.pop(file_hash, None)
    assert first.status == "indexed"



def test_hybrid_retrieval_with_real_chroma(real_store, monkeypatch):
    from langchain_core.documents import Document
    from app import retriever

    doc = Document(
        page_content="Attendance eligibility requires a minimum of 75 percent.",
        metadata={"document_id": "hybrid-doc", "source": "attendance.txt", "page": 0, "chunk_id": "hybrid-chunk"},
    )
    vector_store.add_chunks([doc], ["hybrid-chunk"])
    monkeypatch.setattr(retriever, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(retriever, "USE_MMR", False)
    monkeypatch.setattr(retriever, "MIN_RELEVANCE_SCORE", 0.0)
    results = retriever.retrieve("minimum attendance eligibility 75 percent", k=3)
    assert results
    assert results[0][0].metadata["chunk_id"] == "hybrid-chunk"
