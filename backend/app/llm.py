from langchain_openai import ChatOpenAI

from .config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


_llm = None


def get_llm() -> ChatOpenAI:
    global _llm

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    if _llm is None:
        _llm = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            temperature=0,
        )

    return _llm
