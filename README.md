# AI Document Intelligence

A full-stack Retrieval-Augmented Generation (RAG) application for uploading PDF/TXT files, indexing their text, and asking grounded questions with source-page references. The UI is React + Vite + Tailwind; the API is FastAPI; retrieval uses LangChain and ChromaDB; embeddings use Hugging Face `sentence-transformers/all-MiniLM-L6-v2`; answers use a Groq-hosted model through LangChain's OpenAI-compatible `ChatOpenAI` client.

## Features

- PDF/TXT upload with streamed 10 MB limit enforcement, extension/content checks, and robust TXT decoding.
- SHA-256 duplicate detection with a synchronized JSON document registry.
- Persistent Chroma vector storage using cosine distance.
- Configurable chunk size, overlap, Top-K, context budget, and optional MMR retrieval.
- Follow-up question rewriting from up to six recent chat messages.
- JSON query endpoint and SSE token streaming endpoint with a frontend JSON fallback.
- Markdown-rendered assistant responses, source citations, per-document page/chunk counts, and clear-chat UX.
- Prompt-injection-resistant instructions treating retrieved document text as untrusted data.
- Retrieval/LLM/total timing logs; health reports API key configuration as a boolean and vector-store reachability.
- Docker Compose backend + production Nginx frontend, persistent data, and Hugging Face model cache volume.

## Architecture

```text
React/Vite UI
  ├─ upload/list/delete documents ──> FastAPI
  └─ chat + history ────────────────> POST /query or POST /query/stream
                                      │
                         document loader + splitter
                                      │
                         Hugging Face embeddings
                                      │
                               ChromaDB (cosine)
                                      │
                      retrieval / optional MMR / context
                                      │
                      Groq ChatOpenAI (answer or tokens)
                                      │
                           answer + source metadata
```

`DATA_DIR` contains `uploaded_docs/`, `chroma_db/`, and `document_registry.json`. The registry keeps list requests independent of a full Chroma chunk scan.

## Run locally

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env
# Set GROQ_API_KEY in .env before asking questions
uvicorn app.main:app --reload
```

The API is at `http://localhost:8000`; Swagger is at `/docs`. `DATA_DIR` defaults to the repository-root `data/` directory when started from the backend directory. Override it with an absolute path if preferred.

### Frontend

```bash
cd frontend
cp .env.example .env
npm install --package-lock-only --ignore-scripts  # refresh lock after dependency changes
npm ci
npm run dev
```

The Vite development server defaults to `http://localhost:5173`. `VITE_API_URL` defaults to `http://localhost:8000`.

## Configuration

Copy `backend/.env.example` to `backend/.env` for local backend development. The frontend has `frontend/.env.example`. Important backend variables:

| Variable | Default | Purpose |
|---|---:|---|
| `GROQ_API_KEY` | unset | Groq API credential; never commit it |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `DATA_DIR` | `./data` at project root | Persistent uploads, vector DB, registry |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Text splitting settings |
| `TOP_K` | `6` | Retrieved chunks |
| `MAX_CONTEXT_CHARS` | `12000` | Context character budget |
| `MAX_UPLOAD_MB` | `10` | Upload cap; enforced while reading |
| `MIN_RELEVANCE_SCORE` | `0.0` | Cosine relevance threshold; zero disables filtering |
| `USE_MMR` | `false` | Enable MMR retrieval |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status, Groq-key-configured boolean, vector-store reachability |
| POST | `/documents/upload` | Upload and index PDF/TXT |
| GET | `/documents` | Registry-backed document list, including pages and chunks |
| DELETE | `/documents/{document_id}` | Remove vector chunks, uploaded file, and registry record |
| POST | `/query` | Standard JSON RAG query |
| POST | `/query/stream` | Server-Sent Events: `token` events followed by `final` sources |

Query example:

```json
{
  "question": "What is the attendance requirement?",
  "document_id": null,
  "top_k": 6,
  "history": [
    {"role": "user", "content": "Summarize the attendance policy."},
    {"role": "assistant", "content": "It describes minimum attendance."}
  ]
}
```

`history` is optional and limited to six messages. `top_k` must be 1–20. The response includes `answer`, `sources`, and `retrieval_latency_seconds`. Source pages are one-indexed in both the context header and API response; stored Chroma page metadata remains zero-indexed.

## Tests

Tests use deterministic fake embeddings and a fake LLM; they do not require a Groq key or model download.

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q

cd ../frontend
npm ci
npm test
npm run build
```

## Evaluation

See [`eval/README.md`](eval/README.md). Replace the placeholder dataset with real questions and expected filenames/pages from your own corpus. Run the API evaluation with:

```bash
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6
```

The report contains Hit@K, MRR, mean retrieval latency, answer keyword coverage, and out-of-scope fallback accuracy. The optional local chunk-size sweep re-indexes local documents at 500/700/1000 characters and **deletes/rebuilds the local Chroma collection and registry**; back up `data/` first. No benchmark scores are included here because no user-specific evaluation corpus has been measured.

## Docker

From the repository root, copy `backend/.env.example` to `.env`, set `GROQ_API_KEY`, then run:

```bash
docker compose up --build
```

The production frontend is served at `http://localhost:8080`; the API is at `http://localhost:8000`. Compose mounts `./data` at `/app/data`, uses a named Hugging Face cache volume, and waits for the backend healthcheck before starting the frontend. The local Vite workflow remains available with `npm run dev`.

## Design decisions and trade-offs

- **Chunk size (700) and overlap (100):** a practical starting point that preserves nearby context without creating excessive overlap. The right setting depends on document layout and should be selected with the included evaluation script, not assumed to be universally optimal.
- **Cosine vs. L2:** cosine is configured explicitly because this application compares embedding direction and uses Chroma's normalized relevance-score API. Existing Chroma collections created with the old default distance metric must be re-indexed: stop the app, delete `data/chroma_db/`, and upload the documents again. Keep a backup if the old index matters.
- **Top-K (6):** limits prompt size and latency while allowing multiple passages. Larger values can add noise and consume more context; tune using Hit@K/MRR and answer quality on your own corpus.
- **Why Groq:** it provides hosted inference through an OpenAI-compatible endpoint, so LangChain's `ChatOpenAI` client can be used without a provider-specific chain. It introduces network dependency, rate limits, and key management.
- **Why Chroma:** it is a lightweight persistent vector store suitable for a local/single-service project. At roughly 10,000 documents (and potentially far more chunks), ingestion, metadata filtering, memory, concurrent writes, backup/recovery, and query latency need load testing; a managed or distributed vector database and background ingestion workers may become appropriate. A document count alone does not predict chunk count.
- **Duplicate-upload race handling:** a process-wide re-entrant lock covers hash lookup, indexing, and registry write, so concurrent uploads in the same backend process cannot both index identical content. Multi-process/multi-host deployments need a cross-process lock or transactional database uniqueness constraint.
- **MMR:** optional and off by default. It trades a little retrieval work for diversity among returned chunks. Hybrid BM25 + vector search was deliberately not added because a robust corpus-wide sparse index, synchronization, and score fusion would increase complexity beyond the reliability of this change set.
- **Registry:** JSON is intentionally small and simple for this single-instance project. For multi-worker deployments or frequent writes, migrate it to SQLite/PostgreSQL with transactional uniqueness and locking.

## Known limitations

- PDFs are text-extracted, not OCR-processed. Scanned/image-only PDFs return HTTP 422 with a clear extractable-text error; OCR is intentionally not bundled.
- Embedding model initialization requires the model to be available locally or downloadable on first use. The Hugging Face Docker volume caches it across container recreation.
- Groq queries require internet access and a valid API key. Backend tests mock the LLM and embeddings and do not validate provider availability.
- JSON registry locking is process-local; it is not a distributed lock.
- MMR is implemented; hybrid BM25/vector retrieval is not implemented.
- Streaming fallback can retry the standard endpoint if the stream fails, which may repeat inference if the provider completed a response but the connection dropped before the final event.
- The example evaluation dataset contains illustrative placeholders only. Metrics are meaningful only after replacing it with labeled questions from your own documents.
- Existing cosine/L2 collections are not migrated automatically; re-index them after changing the metric.

## Project structure

```text
backend/              FastAPI app, requirements, pytest suite
frontend/             React/Vite app, components, Vitest suite
  Dockerfile          Production Nginx build
  nginx.conf          SPA fallback routing
eval/                 Placeholder dataset and reproducible evaluation
docs/                 Interview preparation notes
.github/workflows/    CI pipeline
data/                 Runtime data (not committed or packaged)
docker-compose.yml
```
