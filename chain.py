from embeddings_vectorstore import load_vector_store
from retriever import get_relevant_docs
from prompt_llm import generate_answer
from collections import Counter


def create_chain():

    vector_db = load_vector_store()

    def run_chain(question):

        docs = get_relevant_docs(vector_db, question)

        if not docs:
            return generate_answer([], question, [])

        sources = [doc.metadata.get("source") for doc in docs]

        source_count = Counter(sources)

        most_common_source, freq = source_count.most_common(1)[0]

        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("source") == most_common_source
        ]

        pages = sorted(
            list({doc.metadata.get("page", 0) + 1 for doc in filtered_docs})
        )

        return generate_answer(filtered_docs, question, pages)

    return run_chain
