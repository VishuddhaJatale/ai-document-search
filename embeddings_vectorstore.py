from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, VECTOR_DIR
import os
import shutil

def create_vector_store(chunks):

    if os.path.exists(VECTOR_DIR):
        try:
            shutil.rmtree(VECTOR_DIR)
        except:
            pass

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )


def load_vector_store():

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )
