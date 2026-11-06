from langchain_core.documents import Document
import pytest

from app.document_service import _file_hash, _safe_filename
from app.rag import _build_context
from app.schemas import QueryRequest
from app.text_splitter import split_documents


def test_safe_filename():
    assert _safe_filename("../../my file?.pdf") == "my_file_.pdf"


def test_file_hash_is_sha256():
    assert len(_file_hash(b"hello")) == 64
    assert _file_hash(b"hello") == _file_hash(b"hello")


def test_split_documents_respects_chunk_size(monkeypatch):
    import app.text_splitter as splitter_module
    monkeypatch.setattr(splitter_module, "CHUNK_SIZE", 80, raising=False)
    monkeypatch.setattr(splitter_module, "CHUNK_OVERLAP", 10, raising=False)
    docs = split_documents([Document(page_content="word " * 100)])
    assert docs
    assert all(len(doc.page_content) <= 80 for doc in docs)


def test_build_context_respects_max_chars(monkeypatch):
    import app.rag as rag_module
    monkeypatch.setattr(rag_module, "MAX_CONTEXT_CHARS", 100)
    context = _build_context([(Document(page_content="x" * 500, metadata={"source": "a.txt", "page": 0}), 0.8)])
    assert len(context) <= 100


@pytest.mark.parametrize("payload", [{"question": "valid", "top_k": 0}, {"question": "valid", "top_k": 21}, {"question": "   "}])
def test_query_schema_rejects_invalid_values(payload):
    with pytest.raises(Exception):
        QueryRequest(**payload)


def test_context_page_is_one_indexed_and_metadata_remains_zero_indexed():
    doc = Document(page_content="page content", metadata={"source": "policy.pdf", "page": 0})
    context = _build_context([(doc, 0.9)])
    assert "Page 1" in context
    assert doc.metadata["page"] == 0


def test_decode_text_falls_back_through_encodings():
    from app.document_loader import decode_text
    assert decode_text("héllo".encode("utf-8")) == "héllo"
    assert decode_text("café".encode("cp1252")) == "café"
    assert isinstance(decode_text(b"\x81\x8d\x8f"), str)


def test_unreadable_pdf_raises_value_error(tmp_path):
    from app.document_loader import load_document
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(ValueError, match="No extractable text found"):
        load_document(bad)
