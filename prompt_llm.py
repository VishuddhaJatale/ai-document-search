import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY
from parser import AnswerOutput
from memory import add_to_memory
from config import LLM_MODEL


MAX_CONTEXT = 12000


def generate_answer(docs, question, pages):

    if not docs:
        return AnswerOutput(
            answer="❌ I could not find this information in the uploaded documents.",
            pages=[]
        )

    context = "\n\n".join([doc.page_content for doc in docs])

    if len(context.strip()) < 50:
        return AnswerOutput(
            answer="❌ I could not find this information in the uploaded documents.",
            pages=[]
        )

    context = context[:10000]

    api_key = GOOGLE_API_KEY

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant.

IMPORTANT RULES:
- Use ONLY the given document context
- DO NOT add external information
- If multiple documents are present, summarize EACH document separately
- Use headings with document name if available

FORMAT:

Document 1:
- Summary points

Document 2:
- Summary points

If information is missing say:
"I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    try:
        response = llm.invoke(formatted_prompt)
        answer_text = response.content

    except Exception as e:
        print("GEMINI ERROR:", e)

        return AnswerOutput(
        answer=f"❌ Gemini Error: {str(e)}",
        pages=[]
    )


    parsed = AnswerOutput(
        answer=answer_text,
        pages=pages
    )

    parsed.docs = docs

    add_to_memory(question, answer_text)

    return parsed
