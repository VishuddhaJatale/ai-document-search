from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from config import LLM_MODEL
from parser import AnswerOutput
from memory import add_to_memory, get_memory_text

load_dotenv()


def generate_answer(docs, question, pages):

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI assistant. Answer ONLY using the provided document content.

Conversation History:
{get_memory_text()}

Document Content:
{context_text}

Question:
{question}

If answer is not found in document, say "Not found in document".
"""

    response = llm.invoke(prompt)

    parsed = AnswerOutput(
        answer=response.content.strip(),
        pages=pages
    )

    add_to_memory(question, parsed.answer)

    return parsed
