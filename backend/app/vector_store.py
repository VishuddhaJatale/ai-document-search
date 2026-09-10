from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from .config import VECTOR_DIR
from .embeddings import get_embeddings

COLLECTION_NAME = "document_chunks"


_vector_store: Optional[Chroma] = None


def get_vector_store() -> Chroma:
    global _vector_store

    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(VECTOR_DIR),
            embedding_function=get_embeddings(),
        )

    return _vector_store


def add_chunks(chunks: List[Document], ids: List[str]) -> None:
    clean_chunks = filter_complex_metadata(chunks)
    get_vector_store().add_documents(clean_chunks, ids=ids)


def find_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Return metadata for an already indexed file, if its SHA-256 exists."""
    store = get_vector_store()
    result = store.get(where={"file_hash": file_hash}, limit=1)

    metadatas = result.get("metadatas", [])
    if not metadatas:
        return None

    return metadatas[0]

def list_documents() -> List[Dict[str, Any]]:
    """Return unique indexed documents based on stored chunk metadata."""
    store = get_vector_store()

    result = store.get()

    metadatas = result.get("metadatas", [])

    documents = {}

    for metadata in metadatas:
        document_id = metadata.get("document_id")

        if not document_id:
            continue

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata.get("source", ""),
                "pages": int(metadata.get("page_count", 0)),
                "file_hash": metadata.get("file_hash", ""),
            }

    return list(documents.values())

def delete_document(document_id: str) -> bool:
    store = get_vector_store()
    result = store.get(where={"document_id": document_id})
    ids = result.get("ids", [])

    if not ids:
        return False

    store.delete(ids=ids)
    return True
