from langchain_core.documents import Document

from app import rag


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["vector_store_reachable"] is True
    assert isinstance(response.json()["groq_api_key_configured"], bool)


def test_upload_txt_duplicate_and_list(client):
    payload = {"file": ("policy.txt", b"Attendance policy is 75 percent.", "text/plain")}
    first = client.post("/documents/upload", files=payload)
    assert first.status_code == 200
    assert first.json()["chunks"] >= 1
    second = client.post("/documents/upload", files=payload)
    assert second.status_code == 200
    assert second.json()["status"] == "already_exists"
    listed = client.get("/documents")
    assert listed.status_code == 200
    assert listed.json()[0]["chunks"] >= 1


def test_upload_validation(client):
    assert client.post("/documents/upload", files={"file": ("empty.txt", b"", "text/plain")}).status_code == 400
    assert client.post("/documents/upload", files={"file": ("wrong.doc", b"data", "application/msword")}).status_code == 400
    assert client.post("/documents/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}).status_code == 400
    scanned = client.post("/documents/upload", files={"file": ("scanned.pdf", b"%PDF-1.4\nnot a valid text PDF", "application/pdf")})
    assert scanned.status_code == 422
    assert "No extractable text found" in scanned.json()["detail"]
    assert client.post("/documents/upload", files={"file": ("big.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain")}).status_code == 413


def test_non_utf8_txt_is_accepted(client):
    response = client.post("/documents/upload", files={"file": ("legacy.txt", "caf\xe9".encode("cp1252"), "text/plain")})
    assert response.status_code == 200


def test_query_returns_answer_and_sources(client, monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda **_kwargs: [(Document(page_content="75 percent", metadata={"document_id": "d1", "source": "policy.txt", "page": 0, "chunk_id": "c1"}), 0.9)])
    response = client.post("/query", json={"question": "What is the attendance requirement?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "A deterministic test answer."
    assert response.json()["sources"][0]["page"] == 1


def test_query_fallback_has_no_sources(client, monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda **_kwargs: [])
    response = client.post("/query", json={"question": "Unknown?"})
    assert response.json()["answer"] == rag.FALLBACK_ANSWER
    assert response.json()["sources"] == []


def test_document_filter_is_passed_to_retrieval(client, monkeypatch):
    seen = {}
    def fake_retrieve(**kwargs):
        seen.update(kwargs)
        return [(Document(page_content="doc", metadata={"document_id": "d2", "source": "b.txt", "page": 0, "chunk_id": "c2"}), 0.8)]
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    response = client.post("/query", json={"question": "question", "document_id": "d2"})
    assert response.status_code == 200
    assert seen["document_id"] == "d2"
    assert {s["document_id"] for s in response.json()["sources"]} == {"d2"}


def test_delete_removes_file_and_registry(client, tmp_path, monkeypatch):
    from app import document_service
    uploaded = client.post("/documents/upload", files={"file": ("delete.txt", b"some document content", "text/plain")}).json()
    doc_id = uploaded["document_id"]
    registry = document_service.REGISTRY_PATH
    assert registry.exists()
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert not list(document_service.UPLOAD_DIR.glob(f"{doc_id}_*"))
    assert all(item["document_id"] != doc_id for item in __import__("json").loads(registry.read_text()))
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_stream_endpoint_emits_tokens_and_final_sources(client, monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda **_kwargs: [(Document(page_content="test", metadata={"document_id": "d3", "source": "stream.txt", "page": 0, "chunk_id": "c3"}), 0.7)])
    response = client.post("/query/stream", json={"question": "Stream this answer"})
    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: final" in response.text
    assert "stream.txt" in response.text


def test_registry_reindexes_when_vector_index_is_missing(client, isolated_data):
    payload = {"file": ("rebuild.txt", b"Rebuild this document after clearing Chroma.", "text/plain")}
    first = client.post("/documents/upload", files=payload).json()
    isolated_data.clear()  # Simulate deleting/recreating the Chroma collection.
    second = client.post("/documents/upload", files=payload).json()
    assert second["status"] == "indexed"
    assert second["document_id"] != first["document_id"]
    assert second["chunks"] >= 1
