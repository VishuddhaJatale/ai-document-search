def generate_answer(docs, question, pages):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant answering questions from document content only.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer clearly.
"""
    )

    formatted_prompt = prompt.format(
        history=get_memory_text(),
        context="\n".join([doc.page_content for doc in docs]),
        question=question
    )

    response = llm.invoke(formatted_prompt)

    parsed = AnswerOutput(
        answer=response.content,
        pages=pages
    )

    add_to_memory(question, parsed.answer)

    return parsed
