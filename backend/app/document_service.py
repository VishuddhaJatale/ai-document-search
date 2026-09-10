import hashlib
import re
import uuid
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from .config import UPLOAD_DIR
from .document_loader import load_document
from .schemas import DocumentResponse
from .text_splitter import split_documents
from .vector_store import (
    add_chunks,
    delete_document,
    find_document_by_hash,
    list_documents as list_vector_documents,
)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _add_metadata(
    documents: List[Document],
    document_id: str,
    filename: str,
    file_hash: str,
    page_count: int,
) -> None:
    for doc in documents:
        doc.metadata = {
            "document_id": document_id,
            "source": filename,
            "page": int(doc.metadata.get("page", 0)),
            "file_hash": file_hash,
            "page_count": page_count,
        }


def index_document(filename: str, content: bytes) -> DocumentResponse:
    """Persist and index a document, avoiding duplicate content by SHA-256."""
    safe_name = _safe_filename(filename)
    file_hash = _file_hash(content)

    # Deduplication happens before creating a new UUID or writing a new file.
    existing = find_document_by_hash(file_hash)
    if existing:
        return DocumentResponse(
            document_id=existing.get("document_id", ""),
            filename=existing.get("source", safe_name),
            pages=int(existing.get("page_count", 0)),
            chunks=0,
            status="already_exists",
        )

    document_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{document_id}_{safe_name}"
    file_path.write_bytes(content)

    try:
        documents = load_document(file_path)
        if not documents:
            raise ValueError("No document content could be extracted")

        page_count = len(documents)
        _add_metadata(documents, document_id, safe_name, file_hash, page_count)
        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("Document could not be split into chunks")

        chunk_ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{index}"
            chunk.metadata["chunk_id"] = chunk_id
            chunk_ids.append(chunk_id)

        add_chunks(chunks, chunk_ids)

        return DocumentResponse(
            document_id=document_id,
            filename=safe_name,
            pages=page_count,
            chunks=len(chunks),
            status="indexed",
        )

    except Exception:
        file_path.unlink(missing_ok=True)
        raise

def list_documents() -> List[DocumentResponse]:
    """Return all indexed documents."""
    documents = list_vector_documents()

    return [
        DocumentResponse(
            document_id=document["document_id"],
            filename=document["filename"],
            pages=document["pages"],
            chunks=0,
            status="indexed",
        )
        for document in documents
    ]

def remove_document(document_id: str) -> bool:
    deleted = delete_document(document_id)

    for file_path in UPLOAD_DIR.glob(f"{document_id}_*"):
        file_path.unlink(missing_ok=True)

    return deleted
