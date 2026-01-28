from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from config import EMBEDDING_MODEL

# Use in-memory vector DB (Cloud safe)
VECTOR_DB = None


def create_vector_store(chunks):
    global VECTOR_DB

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    clean_chunks = filter_complex_metadata(chunks)

    VECTOR_DB = Chroma.from_documents(
        documents=clean_chunks,
        embedding=embeddings
    )


def load_vector_store():
    global VECTOR_DB
    return VECTOR_DB
