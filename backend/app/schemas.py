from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
