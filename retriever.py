def get_relevant_docs(vector_db, question, k=10):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(question)
