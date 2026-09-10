from embeddings_vectorstore import load_vector_store
from retriever import get_relevant_docs
from prompt_llm import generate_answer


def create_chain():

    vector_db = load_vector_store()

    def run_chain(question):

        docs = get_relevant_docs(vector_db, question)

        if not docs:
            return generate_answer([], question, [])

        pages = sorted(
            list({doc.metadata.get("page", 0) + 1 for doc in docs})
        )

        return generate_answer(docs, question, pages)

    return run_chain
