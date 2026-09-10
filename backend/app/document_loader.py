from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        try:
            docs = PyPDFLoader(str(file_path)).load()
            if docs:
                return docs
        except Exception:
            pass

        return UnstructuredPDFLoader(
            str(file_path), strategy="hi_res"
        ).load()

    if suffix == ".txt":
        return TextLoader(str(file_path), encoding="utf-8").load()

    raise ValueError(f"Unsupported file type: {suffix}")
