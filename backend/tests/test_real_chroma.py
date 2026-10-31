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
