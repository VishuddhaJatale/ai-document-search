import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
UPLOAD_DIR = DATA_DIR / "uploaded_docs"
VECTOR_DIR = DATA_DIR / "chroma_db"
REGISTRY_PATH = DATA_DIR / "document_registry.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
# Chroma cosine relevance scores are normalized to [0, 1]. Keep filtering off by default.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.0"))
USE_MMR = os.getenv("USE_MMR", "false").lower() in {"1", "true", "yes"}
CORS_ORIGINS = [v.strip() for v in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if v.strip()]

for directory in (UPLOAD_DIR, VECTOR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Max number of documents being embedded at the same time. Embedding is CPU
# heavy, so a small cap overlaps uploads without thrashing the CPU or RAM.
INDEXING_CONCURRENCY = max(1, int(os.getenv("INDEXING_CONCURRENCY", "2")))

# Load the embedding model in the background when the API starts.
WARMUP_EMBEDDINGS = os.getenv("WARMUP_EMBEDDINGS", "1").lower() in {"1", "true", "yes"}
