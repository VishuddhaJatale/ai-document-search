from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import TOP_K
from .document_service import (
    index_document,
    list_documents,
    remove_document,
)
from .rag import answer_question
from .schemas import DeleteResponse, DocumentResponse, QueryRequest, QueryResponse


app = FastAPI(
    title="AI Document Intelligence API",
    version="1.0.0",
    description="RAG-based document search and question answering API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return index_document(filename, content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        return answer_question(
            question=request.question,
            top_k=request.top_k or TOP_K,
            document_id=request.document_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/documents", response_model=list[DocumentResponse])
def get_documents():
    try:
        return list_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
    
@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    deleted = remove_document(document_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return DeleteResponse(document_id=document_id, deleted=True)
