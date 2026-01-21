from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from config import EMBEDDING_MODEL

CHROMA_PATH = "data/chroma_db"


def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    clean_chunks = filter_complex_metadata(chunks)

    Chroma.from_documents(
        clean_chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
