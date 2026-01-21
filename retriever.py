import re

def get_relevant_docs(vector_db, question, k=3):

    match = re.search(r"page\s*(\d+)", question.lower())

    if match:

        page_number = int(match.group(1)) - 1

        docs = vector_db.get()["documents"]
        metas = vector_db.get()["metadatas"]

        result_docs = []

        for text, meta in zip(docs, metas):
            if meta.get("page") == page_number:
                result_docs.append(
                    type("Doc", (), {"page_content": text, "metadata": meta})
                )

        return result_docs

    retriever = vector_db.as_retriever(search_kwargs={"k": k})

    return retriever.invoke(question)
