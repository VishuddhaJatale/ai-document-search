# AI Document Intelligence

[![CI](https://github.com/VishuddhaJatale/ai-document-search/actions/workflows/ci.yml/badge.svg)](https://github.com/VishuddhaJatale/ai-document-search/actions/workflows/ci.yml)

A full-stack Retrieval-Augmented Generation (RAG) application. Upload PDF or TXT files, index their text, and ask questions that are answered **only from your documents**, with source page references.

**Stack:** React + Vite + Tailwind (UI) · FastAPI (API) · LangChain + ChromaDB (retrieval) · Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (embeddings) · Groq-hosted LLM via LangChain's OpenAI-compatible `ChatOpenAI` client (answers).

## Features

- PDF/TXT upload with a streamed 10 MB limit, extension and content checks, and robust TXT decoding (UTF-8, then cp1252, then latin-1).
- SHA-256 duplicate detection with a synchronized JSON document registry. Concurrent identical uploads are safe: one request indexes, the others report `already_exists`.
- Persistent Chroma vector storage using cosine distance.
- Configurable chunk size, overlap, Top-K, context budget, optional relevance threshold and optional MMR retrieval.
- Follow-up question rewriting using up to six recent chat messages.
- JSON query endpoint, SSE token-streaming endpoint (with a frontend JSON fallback), and a retrieval-only `/retrieve` endpoint for debugging and evaluation.
- Grounded answers: the model is instructed to refuse when the documents do not contain the answer, and retrieved text is treated as untrusted data (prompt-injection-resistant instructions).
- Markdown-rendered answers, source citations, per-document page and chunk counts, and clear-chat UX.
- Retrieval, LLM and total timing logs. `/health` reports the API key as a boolean and vector-store reachability.
- Docker Compose setup: backend, production Nginx frontend, persistent data, and a Hugging Face model cache volume.
- Automated tests (backend and frontend) and CI on every push.

## Architecture

```text
React/Vite UI
  ├─ upload / list / delete documents ──> FastAPI
  └─ chat + history ────────────────────> POST /query or POST /query/stream
                                          │
                             document loader + text splitter
                                          │
                             Hugging Face embeddings (local)
                                          │
                                   ChromaDB (cosine)
                                          │
                          retrieval / optional MMR / context
                                          │
                          Groq ChatOpenAI (answer or tokens)
                                          │
                               answer + source metadata
```

`DATA_DIR` contains `uploaded_docs/`, `chroma_db/` and `document_registry.json`. The registry keeps document-list requests independent of a full Chroma scan.

## Run locally

Prerequisites: Python 3.11+, Node.js 20+ (CI uses 22), and a free [Groq](https://console.groq.com) API key.

### Backend

```bash
cd backend
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000` and Swagger docs are at `/docs`. The first run downloads the embedding model, so the first upload can be slow. `DATA_DIR` defaults to a `data/` folder at the repository root; override it with an absolute path if you prefer.

### Frontend

```bash
cd frontend
cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env
npm ci
npm run dev
```

The UI runs at `http://localhost:5173`. `VITE_API_URL` defaults to `http://localhost:8000`.

## Configuration

Copy `backend/.env.example` to `backend/.env` for local development. Main backend variables:

| Variable | Default | Purpose |
|---|---:|---|
| `GROQ_API_KEY` | unset | Groq API credential. Never commit it |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model name (check it is available on your account) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `DATA_DIR` | `./data` at project root | Uploads, vector DB and registry |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Text splitting |
| `TOP_K` | `6` | Retrieved chunks per question |
| `MAX_CONTEXT_CHARS` | `12000` | Context character budget |
| `MAX_UPLOAD_MB` | `10` | Upload cap, enforced while reading |
| `MIN_RELEVANCE_SCORE` | `0.0` | Per-chunk cosine relevance threshold. `0` disables filtering |
| `USE_MMR` | `false` | Enable MMR retrieval |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8080` | Comma-separated allowed origins |
| `ANONYMIZED_TELEMETRY` | `False` | Chroma usage telemetry, off by default |

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status, API-key-configured boolean, vector-store reachability |
| POST | `/documents/upload` | Upload and index a PDF or TXT file |
| GET | `/documents` | Registry-backed document list with pages and chunks |
| DELETE | `/documents/{document_id}` | Remove vector chunks, uploaded file and registry record |
| POST | `/retrieve` | Retrieval only (no LLM call): ranked sources for a question. Used for debugging and retrieval evaluation |
| POST | `/query` | Standard JSON RAG query |
| POST | `/query/stream` | Server-Sent Events: `token` events followed by a `final` event with sources |

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

`question` must be 1–4000 characters. `history` is optional and limited to six messages. `top_k` must be 1–20. The response contains `answer`, `sources` and `retrieval_latency_seconds`. Source pages are one-indexed in the API response; stored Chroma page metadata is zero-indexed.

## Tests and CI

Tests use deterministic fake embeddings and a fake LLM, so they need no Groq key and no model download.

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q                 # 25 tests

cd ../frontend
npm ci
npm run lint
npm test                  # 4 tests
npm run build
```

Backend tests include API behaviour, unit tests, and tests against a real Chroma database, including a regression test for concurrent identical uploads. GitHub Actions runs the backend tests and the frontend lint, tests and build on every push and pull request.

## Evaluation

The repository includes a labelled dataset (`eval/dataset.json`: 33 in-scope and 6 out-of-scope questions over three documents) and a reproducible runner. See [`eval/README.md`](eval/README.md).

```bash
# Retrieval only: fast, free, no LLM calls
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6 --skip-answers

# Retrieval + answers
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6
```

Hit@K and MRR are computed from the `/retrieve` endpoint, so they measure the retriever only.

### Baseline results

Default configuration: chunk size 700, overlap 100, Top-K 6, `all-MiniLM-L6-v2`, dense retrieval only.

| Metric | Result |
|---|---|
| Hit@6 (PDF questions) | 24 / 27 = 88.9% |
| MRR (PDF questions) | 0.69 |
| Answer keyword coverage | 34 / 39 = 87.2% |
| Out-of-scope refusal | 6 / 6 = 100% |
| Mean retrieval latency | about 42 ms |

Per document: `week_6_Logic-Programming.pdf` 13/13 hits (MRR 0.96); `15._KRR-CD-Theory.pdf`, a long slide deck, 11/14 hits (MRR 0.43).

How to read these numbers:

- Retrieval metrics exclude the TXT file. Its chunks all carry page 1, so a page-based label cannot tell chunks apart and would inflate the score.
- The dataset is small (33 in-scope questions), so one question moves Hit@6 by about 3 points.
- Hit@K uses strict page labels. Some misses are adjacent-slide near-misses that still produced a correct answer.
- The keyword check is a crude substring match. At least one miss is a false negative ("terminate" vs "termination"). The remaining misses trace to retrieval failures on the KRR deck, and the model refused instead of guessing.

### Chunk-size sweep

An optional local sweep re-indexes at 500/700/1000 characters and **deletes and rebuilds the local Chroma collection and registry**. Back up `data/` first.

## Docker

From the repository root, create a `.env` file next to `docker-compose.yml` containing `GROQ_API_KEY=...` (Docker Compose does not read `backend/.env`), then run:

```bash
docker compose up --build
```

The frontend is served at `http://localhost:8080` and the API at `http://localhost:8000`. Compose mounts `./data` at `/app/data`, uses a named Hugging Face cache volume, and waits for the backend healthcheck before starting the frontend.

## Design decisions and trade-offs

- **Chunk size (700) and overlap (100):** a practical starting point that keeps nearby context without heavy duplication. The best value depends on document layout and should be chosen with the evaluation script.
- **Cosine vs. L2:** cosine is configured explicitly because the app compares embedding direction and uses Chroma's normalized relevance-score API. Collections created with the old default metric must be re-indexed: stop the app, delete `data/chroma_db/`, and upload the documents again.
- **Top-K (6):** limits prompt size and latency while allowing several passages. Larger values add noise and cost; tune with Hit@K/MRR and answer quality.
- **Why Groq:** hosted inference through an OpenAI-compatible endpoint, so LangChain's `ChatOpenAI` client works without a provider-specific chain. The trade-offs are network dependency, rate limits and key management.
- **Why Chroma:** a lightweight persistent vector store that suits a local, single-service project. At much larger scale, ingestion, filtering, concurrent writes, backup and latency need load testing, and a managed vector database with background ingestion workers may become appropriate.
- **Duplicate-upload handling:** a short process-wide re-entrant lock covers only the hash lookup and registry write. Heavy work (parse, split, embed, store) runs outside the lock. An in-flight map keyed by file hash makes concurrent identical uploads wait for the first one and then report `already_exists`. Chunks of an in-flight upload are never adopted as a legacy record, which previously could create duplicate registry entries. Multi-process deployments need a cross-process lock or a database uniqueness constraint.
- **MMR:** optional and off by default. It adds diversity among returned chunks at the cost of a little extra retrieval work.
- **Registry:** JSON is deliberately simple for a single-instance project. For multiple workers or frequent writes, move it to SQLite or PostgreSQL with transactional uniqueness.

## Known limitations

- **No authentication or rate limiting.** Add both before exposing the API publicly, because anyone could delete documents or consume the Groq quota.
- PDFs are text-extracted with PyPDF, with no OCR. Scanned, image-only or corrupt PDFs return HTTP 422 with a clear message.
- The relevance threshold (`MIN_RELEVANCE_SCORE`) is applied per chunk, not to the best chunk, so a high value can drop useful lower-ranked chunks. It is off by default.
- Retrieval is dense-only. Exact-term queries such as acronyms can rank poorly, and a single summary slide can occupy the top result for several questions.
- The source list in a response contains all retrieved chunks, not only the ones the model used.
- The embedding model must be available locally or downloadable on first use. The Docker volume caches it.
- Groq queries need internet access and a valid key. Backend tests mock the LLM and embeddings and do not validate provider availability.
- Registry locking is process-local, not distributed.
- If a stream fails after the provider finished responding, the frontend fallback can repeat inference.
- Existing cosine/L2 collections are not migrated automatically.

## Roadmap

- Chunk-level evaluation labels (expected text snippets) instead of page labels, and a larger question set
- Hybrid keyword + vector search, reranking and neighbouring-slide expansion, each measured against the baseline above
- Authentication and rate limiting, then a public deployment
- SQLite registry for transactional uniqueness

## Project structure

```text
backend/              FastAPI app, requirements, pytest suite
frontend/             React/Vite app, components, Vitest suite
  Dockerfile          Production Nginx build
  nginx.conf          SPA fallback routing
eval/                 Labelled dataset, example dataset, evaluation runner
docs/                 Interview preparation notes
.github/workflows/    CI pipeline
CHANGELOG.md          Notable changes
data/                 Runtime data (not committed)
docker-compose.yml
```
