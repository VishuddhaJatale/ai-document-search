import os
from dotenv import load_dotenv

load_dotenv()   

LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

UPLOAD_DIR = "data/uploaded_docs"
VECTOR_DIR = "data/chroma_db"
