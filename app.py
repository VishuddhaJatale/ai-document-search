import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from load_documents import load_documents
from text_splitting import split_documents
from embeddings_vectorstore import create_vector_store
from chain import create_chain
from config import UPLOAD_DIR, VECTOR_DIR

load_dotenv()

st.set_page_config("AI Document Search", layout="centered")

st.title("📄 AI-Based Document Search & Knowledge Retrieval")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "processed" not in st.session_state:
    st.session_state.processed = False

uploaded_files = st.file_uploader(
    "Upload Documents (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

if st.button("Process Document"):

    if not uploaded_files:
        st.warning("Upload files first")
        st.stop()

    with st.spinner("Indexing documents..."):

        st.session_state.processed = False
        st.session_state.qa_chain = None

        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR, ignore_errors=True)

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        for f in uploaded_files:
            with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
                out.write(f.read())

        documents = load_documents(UPLOAD_DIR)

        if not documents:
            st.error("No document content found")
            st.stop()

        chunks = split_documents(documents)

        if not chunks:
            st.error("Chunking failed")
            st.stop()

        create_vector_store(chunks)

        st.session_state.qa_chain = create_chain()
        st.session_state.processed = True

    st.success("Documents indexed successfully")

st.divider()
st.subheader("Ask Questions")

question = st.text_input("Enter your question")

if st.button("Ask"):

    if not st.session_state.processed:
        st.warning("Process documents first")

    elif not question.strip():
        st.warning("Enter a question")

    else:
        with st.spinner("Generating answer..."):
            result = st.session_state.qa_chain(question)

        st.markdown("Answer")
        st.write(result.answer)

        if result.docs:
            sources = set()

            for doc in result.docs:
                src = doc.metadata.get("source", "file")
                page = doc.metadata.get("page", 0) + 1
                sources.add(f"{src} - Page {page}")

            st.caption("📄 Source: " + ", ".join(sources))
