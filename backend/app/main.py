import hmac
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import (
    API_KEY, CORS_ORIGINS, MAX_UPLOAD_MB, RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS, REQUIRE_API_KEY, TOP_K, WARMUP_EMBEDDINGS,
)
from .embeddings import warm_up
from .document_service import index_document, list_documents, remove_document
from .rag import answer_question, retrieve_sources, stream_answer
from .schemas import DeleteResponse, DocumentResponse, QueryRequest, QueryResponse, RetrieveResponse
from .vector_store import vector_store_is_reachable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
_rate_lock = threading.Lock()
_request_times = defaultdict(deque)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if WARMUP_EMBEDDINGS:
        threading.Thread(target=warm_up, name="embedding-warmup", daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan, title="AI Document Intelligence API", version="2.1.0", description="RAG-based document search and question answering API.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def protect_api(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path == "/health":
        return await call_next(request)

    if REQUIRE_API_KEY:
        supplied = request.headers.get("X-API-Key", "")
        if not API_KEY or not hmac.compare_digest(supplied, API_KEY):
            return JSONResponse(status_code=401, content={"detail": "A valid API key is required."})

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        bucket = _request_times[client_ip]
        while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)
    return await call_next(request)


@app.get("/health")
def health():
    from .config import GROQ_API_KEY, RETRIEVAL_MODE
    return {
        "status": "ok",
        "groq_api_key_configured": bool(GROQ_API_KEY),
        "vector_store_reachable": vector_store_is_reachable(),
        "retrieval_mode": RETRIEVAL_MODE,
        "api_key_required": REQUIRE_API_KEY,
    }


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"pdf", "txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    chunks, total = [], 0
    while True:
        block = await file.read(min(1024 * 1024, max_bytes + 1 - total))
        if not block:
            break
        total += len(block)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
        chunks.append(block)
    content = b"".join(chunks)
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if suffix == "pdf" and not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded PDF does not have a valid PDF signature.")
    try:
        return await run_in_threadpool(index_document, filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document indexing failed")
        raise HTTPException(status_code=500, detail="Document indexing failed") from exc


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        return answer_question(question=request.question, top_k=request.top_k or TOP_K, document_id=request.document_id, history=request.history)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from exc


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_documents(request: QueryRequest):
    try:
        return retrieve_sources(question=request.question, top_k=request.top_k or TOP_K, document_id=request.document_id)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed") from exc


@app.post("/query/stream")
def query_documents_stream(request: QueryRequest):
    def events():
        try:
            for event, payload in stream_answer(request.question, request.top_k or TOP_K, request.document_id, request.history):
                yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Streaming query failed")
            yield f"event: error\ndata: {json.dumps({'detail': 'Streaming query failed'})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/documents", response_model=list[DocumentResponse])
def get_documents():
    try:
        return list_documents()
    except Exception as exc:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=500, detail="Document listing failed") from exc


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    if not remove_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteResponse(document_id=document_id, deleted=True)
