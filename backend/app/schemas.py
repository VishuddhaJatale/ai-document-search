from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: Optional[float] = None

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    history: List[ChatMessage] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    retrieval_latency_seconds: Optional[float] = None

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str

class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
