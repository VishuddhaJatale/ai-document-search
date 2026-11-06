import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
NO_TEXT_MESSAGE = "No extractable text found, the PDF may be scanned or corrupt"


def decode_text(raw: bytes) -> str:
    """Decode text bytes: UTF-8 first, then cp1252, then latin-1 as a last resort."""
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def load_document(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            docs = PyPDFLoader(str(file_path)).load()
        except Exception as exc:
            logger.warning("PDF could not be parsed: %s", exc)
            raise ValueError(NO_TEXT_MESSAGE) from exc
        if not any(doc.page_content.strip() for doc in docs):
            raise ValueError(NO_TEXT_MESSAGE)
        return docs

    if suffix == ".txt":
        return [Document(page_content=decode_text(file_path.read_bytes()), metadata={"page": 0})]

    raise ValueError(f"Unsupported file type: {suffix}")
