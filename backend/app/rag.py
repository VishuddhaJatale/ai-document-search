from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import MAX_CONTEXT_CHARS, TOP_K
from .llm import get_llm
from .retriever import retrieve
from .schemas import QueryResponse, Source


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document question-answering assistant.

Answer using ONLY the supplied document context.
Do not use outside knowledge.
If the context does not contain enough information to answer the question,
say exactly: "I could not find this information in the uploaded documents."
Do not invent facts, sources, pages, or citations.
Keep the answer clear and concise.

Document context:
{context}
""",
        ),
        ("human", "Question: {question}"),
    ]
)


def _build_context(results: List[Tuple[Document, Optional[float]]]) -> str:
    pieces = []
    total = 0

    for index, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata
        header = (
            f"Source {index}: {metadata.get('source', 'unknown')} | "
            f"Page {metadata.get('page', 0)}"
        )
        piece = f"{header}\n{doc.page_content}"

        if total + len(piece) > MAX_CONTEXT_CHARS:
            break

        pieces.append(piece)
        total += len(piece)

    return "\n\n---\n\n".join(pieces)


def answer_question(
    question: str,
    top_k: int = TOP_K,
    document_id: Optional[str] = None,
) -> QueryResponse:
    results = retrieve(
        question=question,
        k=top_k,
        document_id=document_id,
    )

    if not results:
        return QueryResponse(
            answer="I could not find this information in the uploaded documents.",
            sources=[],
        )

    context = _build_context(results)

    if not context.strip():
        return QueryResponse(
            answer="I could not find this information in the uploaded documents.",
            sources=[],
        )

    messages = PROMPT.format_messages(
        context=context,
        question=question,
    )

    response = get_llm().invoke(messages)
    answer = response.content

    sources = []
    seen = set()

    for doc, score in results:
        metadata = doc.metadata
        key = (
            metadata.get("document_id"),
            metadata.get("page", 0),
            metadata.get("chunk_id"),
        )

        if key in seen:
            continue
        seen.add(key)

        sources.append(
            Source(
                document_id=metadata.get("document_id", ""),
                filename=metadata.get("source", "unknown"),
                page=int(metadata.get("page", 0)) + 1,
                chunk_id=metadata.get("chunk_id", ""),
                score=round(float(score), 4) if score is not None else None,
            )
        )

    return QueryResponse(answer=answer, sources=sources)
