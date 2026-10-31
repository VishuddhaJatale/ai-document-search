import os
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="adi-tests-")
os.environ["DATA_DIR"] = _TEST_DATA_DIR
os.environ.setdefault("GROQ_API_KEY", "")

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import main
from app import document_service, rag, vector_store


class DeterministicFakeEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [[float((len(text) % 7) + 1), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float((len(text) % 7) + 1), 1.0, 0.5]


class FakeLLM:
    def invoke(self, messages):
        return AIMessage(content="A deterministic test answer.")

    def stream(self, messages):
        yield AIMessage(content="A deterministic ")
        yield AIMessage(content="test answer.")


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploaded_docs"
    upload_dir.mkdir(parents=True)
    monkeypatch.setattr(document_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(document_service, "REGISTRY_PATH", data_dir / "document_registry.json")
    indexed_docs = {}
    def fake_find(file_hash):
        return indexed_docs.get(file_hash)
    def fake_add(chunks, _ids):
        metadata = chunks[0].metadata
        indexed_docs[metadata["file_hash"]] = {
            "document_id": metadata["document_id"],
            "source": metadata["source"],
            "page_count": metadata["page_count"],
            "chunks": len(chunks),
        }
    monkeypatch.setattr(document_service, "find_document_by_hash", fake_find)
    monkeypatch.setattr(document_service, "add_chunks", fake_add)
    deleted_ids = set()
    def fake_delete(document_id):
        if document_id in deleted_ids:
            return False
        deleted_ids.add(document_id)
        for file_hash, metadata in list(indexed_docs.items()):
            if metadata["document_id"] == document_id:
                indexed_docs.pop(file_hash)
        return True
    monkeypatch.setattr(document_service, "delete_document", fake_delete)
    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: DeterministicFakeEmbedding())
    monkeypatch.setattr(main, "vector_store_is_reachable", lambda: True)
    yield indexed_docs


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def sample_documents():
    return [Document(page_content="University attendance must be at least 75 percent.", metadata={"page": 0})]

@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding()
