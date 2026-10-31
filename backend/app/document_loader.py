from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            docs = PyPDFLoader(str(file_path)).load()
            if docs and any(doc.page_content.strip() for doc in docs):
                return docs
        except Exception:
            pass
        try:
            docs = UnstructuredPDFLoader(str(file_path)).load()
            if docs and any(doc.page_content.strip() for doc in docs):
                return docs
        except Exception as exc:
            raise ValueError("No extractable text found, the PDF may be scanned") from exc
        raise ValueError("No extractable text found, the PDF may be scanned")

    if suffix == ".txt":
        raw = file_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp1252")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="replace")
        return [Document(page_content=text, metadata={"page": 0})]

    raise ValueError(f"Unsupported file type: {suffix}")
