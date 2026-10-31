import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from .config import INDEXING_CONCURRENCY, REGISTRY_PATH, UPLOAD_DIR
from .document_loader import load_document
from .schemas import DocumentResponse
from .text_splitter import split_documents
from .vector_store import add_chunks, delete_document, find_document_by_hash

logger = logging.getLogger(__name__)
_registry_lock = threading.RLock()   # held only for short registry/dedup steps, never while embedding
_inflight: dict[str, threading.Event] = {}   # file_hash -> event, so identical concurrent uploads index once
_indexing_slots = threading.BoundedSemaphore(INDEXING_CONCURRENCY)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return cleaned or "uploaded_file"


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_registry() -> list[dict]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _write_registry(items: list[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = REGISTRY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    temp_path.replace(REGISTRY_PATH)


def _add_metadata(documents: List[Document], document_id: str, filename: str, file_hash: str, page_count: int) -> None:
    for doc in documents:
        doc.metadata = {
            **doc.metadata,
            "document_id": document_id,
            "source": filename,
            "page": int(doc.metadata.get("page", 0)),
            "file_hash": file_hash,
            "page_count": page_count,
        }


def _find_duplicate_locked(safe_name: str, file_hash: str):
    """Dedup check. Caller must hold _registry_lock. Returns a response if the file already exists."""
    registry = _read_registry()
    existing = next((item for item in registry if item["file_hash"] == file_hash), None)
    existing_vector = find_document_by_hash(file_hash)
    if existing and existing_vector:
        actual_chunks = int(existing_vector.get("chunks", existing["chunks"]))
        if actual_chunks != existing["chunks"]:
            existing["chunks"] = actual_chunks
            _write_registry(registry)
        return DocumentResponse(document_id=existing["document_id"], filename=existing["filename"], pages=existing["pages"], chunks=actual_chunks, status="already_exists")
    if existing and not existing_vector:
        # A registry entry can outlive a manually removed/rebuilt vector DB.
        # Drop the stale record and index this upload again instead of
        # incorrectly reporting a duplicate with no searchable vectors.
        for old_file in UPLOAD_DIR.glob(f"{existing['document_id']}_*"):
            old_file.unlink(missing_ok=True)
        registry = [item for item in registry if item["file_hash"] != file_hash]
        _write_registry(registry)
    # Also detect records indexed before the registry was introduced.
    if existing_vector:
        doc_id = existing_vector.get("document_id", "")
        record = {"document_id": doc_id, "filename": existing_vector.get("source", safe_name), "pages": int(existing_vector.get("page_count", 0)), "chunks": int(existing_vector.get("chunks", 0)), "file_hash": file_hash, "uploaded_at": datetime.now(timezone.utc).isoformat()}
        registry.append(record)
        _write_registry(registry)
        return DocumentResponse(document_id=doc_id, filename=record["filename"], pages=record["pages"], chunks=record["chunks"], status="already_exists")
    return None


def _index_new_document(safe_name: str, file_hash: str, content: bytes) -> DocumentResponse:
    """Heavy work (parse, split, embed, store). Runs WITHOUT holding the registry lock."""
    document_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{document_id}_{safe_name}"
    file_path.write_bytes(content)
    indexed = False
    try:
        documents = load_document(file_path)
        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("No extractable text found, the PDF may be scanned")
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
        indexed = True
        record = {"document_id": document_id, "filename": safe_name, "pages": page_count, "chunks": len(chunks), "file_hash": file_hash, "uploaded_at": datetime.now(timezone.utc).isoformat()}
        with _registry_lock:  # short: re-read the registry so concurrent uploads never overwrite each other
            registry = _read_registry()
            registry.append(record)
            _write_registry(registry)
        return DocumentResponse(document_id=document_id, filename=safe_name, pages=page_count, chunks=len(chunks), status="indexed")
    except Exception:
        if indexed:
            try:
                delete_document(document_id)
            except Exception:
                logger.exception("Failed to roll back partially indexed document")
        file_path.unlink(missing_ok=True)
        raise


def index_document(filename: str, content: bytes) -> DocumentResponse:
    safe_name = _safe_filename(filename)
    file_hash = _file_hash(content)
    while True:
        with _registry_lock:  # short critical section: dedup check + reserve this hash
            duplicate = _find_duplicate_locked(safe_name, file_hash)
            if duplicate:
                return duplicate
            waiting_for = _inflight.get(file_hash)
            if waiting_for is None:
                my_event = threading.Event()
                _inflight[file_hash] = my_event
                break
        # Same content is being indexed by another request: wait, then re-check (it becomes a duplicate).
        waiting_for.wait()
    try:
        with _indexing_slots:  # cap parallel embedding jobs
            return _index_new_document(safe_name, file_hash, content)
    finally:
        with _registry_lock:
            _inflight.pop(file_hash, None)
        my_event.set()


def list_documents() -> List[DocumentResponse]:
    with _registry_lock:
        return [DocumentResponse(document_id=item["document_id"], filename=item["filename"], pages=item["pages"], chunks=item["chunks"], status="indexed") for item in _read_registry()]


def remove_document(document_id: str) -> bool:
    with _registry_lock:
        deleted = delete_document(document_id)
        if not deleted:
            return False
        for file_path in UPLOAD_DIR.glob(f"{document_id}_*"):
            file_path.unlink(missing_ok=True)
        _write_registry([item for item in _read_registry() if item["document_id"] != document_id])
        return True
