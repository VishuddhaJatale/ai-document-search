import threading
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from .config import VECTOR_DIR
from .embeddings import get_embeddings

COLLECTION_NAME = "document_chunks"
_vector_store: Optional[Chroma] = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                _vector_store = Chroma(
                    collection_name=COLLECTION_NAME,
                    persist_directory=str(VECTOR_DIR),
                    embedding_function=get_embeddings(),
                    collection_metadata={"hnsw:space": "cosine"},
                )
    return _vector_store


def add_chunks(chunks: List[Document], ids: List[str]) -> None:
    get_vector_store().add_documents(filter_complex_metadata(chunks), ids=ids)


def find_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    store = get_vector_store()
    result = store.get(where={"file_hash": file_hash}, limit=1)
    metadatas = result.get("metadatas", [])
    if not metadatas:
        return None
    metadata = dict(metadatas[0])
    document_id = metadata.get("document_id")
    if document_id:
        chunks = store.get(where={"document_id": document_id}, include=[]).get("ids", [])
        metadata["chunks"] = len(chunks)
    return metadata


def delete_document(document_id: str) -> bool:
    store = get_vector_store()
    result = store.get(where={"document_id": document_id}, include=[])
    ids = result.get("ids", [])
    if not ids:
        return False
    store.delete(ids=ids)
    return True


def vector_store_is_reachable() -> bool:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        client.heartbeat()
        return True
    except Exception:
        return False
