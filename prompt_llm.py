import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate

from config import LLM_MODEL
from parser import AnswerOutput
from memory import add_to_memory, get_memory_text


def generate_answer(docs, question, pages):

    api_key = st.secrets["GOOGLE_API_KEY"]

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant answering questions strictly using the given document content.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Give a clear and concise answer.
"""
    )

    formatted_prompt = prompt.format(
        history=get_memory_text(),
        context="\n".join([doc.page_content for doc in docs]),
        question=question
    )

    response = llm.invoke(formatted_prompt)

    answer_text = response.content

    parsed = AnswerOutput(
        answer=answer_text,
        pages=pages
    )

    add_to_memory(question, parsed.answer)

    return parsed
