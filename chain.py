from embeddings_vectorstore import load_vector_store
from retriever import get_relevant_docs
from prompt_llm import generate_answer


def create_chain():
    vector_db = load_vector_store()

    def run_chain(question):

        q = question.lower()

        if any(word in q for word in ["summarize", "explain", "everything", "about pdf", "about document"]):

            data = vector_db.get()
            docs = [
                type("Doc", (), {"page_content": text, "metadata": meta})
                for text, meta in zip(data["documents"], data["metadatas"])
            ]
        else:
            docs = get_relevant_docs(vector_db, question)

        pages = sorted(
            list({doc.metadata.get("page", 0) + 1 for doc in docs})
        )
        return generate_answer(docs, question, pages)

    return run_chain
