import streamlit as st
import os

from load_documents import load_documents
from text_splitting import split_documents
from embeddings_vectorstore import create_vector_store
from chain import create_chain

st.set_page_config(
    page_title="AI Document Search System",
    layout="centered"
)

st.title("📄 AI-Based Document Search & Knowledge Retrieval")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader(
    "Upload PDF or Text file",
    type=["pdf", "txt"]
)

if uploaded_file:

    os.makedirs("data/uploaded_docs", exist_ok=True)

    file_path = f"data/uploaded_docs/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded successfully")

    if st.button("Process Document"):

        with st.spinner("Loading and indexing documents..."):

            documents = load_documents("data/uploaded_docs")

            if len(documents) == 0:
                st.error("❌ No document content found to index.")
                st.stop()

            chunks = split_documents(documents)

            if len(chunks) == 0:
                st.error("❌ Text splitting failed. No chunks created.")
                st.stop()

            create_vector_store(chunks)

            st.session_state.qa_chain = create_chain()

        st.success("✅ Documents indexed and chain ready!")

st.divider()
st.subheader("Ask Questions From Document")

question = st.text_input("Enter your question")

if st.button("Ask"):

    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload and process a document first.")

    elif question.strip() == "":
        st.warning("⚠️ Please enter a question.")

    else:
        with st.spinner("Generating answer..."):

            result = st.session_state.qa_chain(question)

        st.markdown("Answer")
        st.write(result.answer)

        if result.pages:
            pages_text = ", ".join(f"Page {p}" for p in result.pages)
            st.caption(f"📄 Source: {pages_text}")
