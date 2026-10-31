import logging
import threading
import time

from .config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embeddings = None
_embeddings_lock = threading.Lock()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                # Import lazily so API/unit tests can inject deterministic embeddings
                # without importing/downloading the heavyweight sentence-transformers stack.
                from langchain_huggingface import HuggingFaceEmbeddings
                _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def warm_up() -> None:
    """Load the model and run one tiny encode so the first real upload is fast."""
    started = time.perf_counter()
    try:
        get_embeddings().embed_query("warm up")
        logger.info("embedding_warmup_done seconds=%.2f", time.perf_counter() - started)
    except Exception:
        logger.exception("embedding_warmup_failed")
