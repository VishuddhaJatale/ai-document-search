from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL


_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
        )

    return _embeddings
