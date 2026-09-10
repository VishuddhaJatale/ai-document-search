# AI Document Intelligence

An AI-powered document search and question-answering application built with **React, FastAPI, LangChain, Hugging Face embeddings, ChromaDB, and Groq**.

The application lets users upload PDF/TXT documents, index their contents as vector embeddings, ask natural-language questions, and receive answers grounded only in the uploaded document context. Answers include source document and page information for traceability.

## Features

- PDF and TXT document upload
- Client-side and backend file validation
- Maximum upload size of 10 MB
- Empty-file validation
- SHA-256 based duplicate-content prevention
- Persistent ChromaDB vector store
- Semantic similarity search
- Configurable Top-K retrieval
- Document-specific or global search
- RAG-based question answering
- Groq LLM through an OpenAI-compatible API
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Source citations with filename and page number
- Persistent uploaded documents
- Document deletion
- Loading and API/network error handling
- React component-based frontend
- FastAPI Swagger/OpenAPI documentation
- Dockerized backend
- Docker volume persistence for uploaded documents and ChromaDB

## Architecture

```text
                         User
                          │
                          ▼
                React + Vite Frontend
                   localhost:5173
                          │
                    HTTP / JSON
                          │
                          ▼
                 FastAPI Backend
                   localhost:8000
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
   Document Pipeline                  Query Pipeline
          │                                │
   PDF / TXT Loader                  User Question
          │                                │
          ▼                                ▼
   Recursive Character              Query Embedding
      Text Splitter                       │
          │                                ▼
          ▼                           ChromaDB
 Hugging Face Embeddings                  │
          │                          Top-K Chunks
          ▼                                │
       ChromaDB                            ▼
          │                         Context Builder
          │                                │
          └───────────────┐                ▼
                          │            Groq LLM
                          │                │
                          └────────────────┤
                                           ▼
                                  Answer + Sources
```

## RAG Pipeline

### 1. Document upload

The frontend sends a PDF or TXT file to:

```text
POST /documents/upload
```

The backend validates the extension and checks that the uploaded file is not empty.

The document service then:

1. Sanitizes the filename.
2. Calculates a SHA-256 hash of the file contents.
3. Checks ChromaDB for an existing file with the same hash.
4. Creates a UUID if the document is new.
5. Saves the uploaded file.
6. Loads the document.
7. Adds document metadata.
8. Splits the document into chunks.
9. Creates chunk IDs.
10. Stores the chunks in ChromaDB.

### 2. Document loading

Supported formats are:

- `.pdf`
- `.txt`

PDFs are loaded using `PyPDFLoader`, with `UnstructuredPDFLoader` used as a fallback. TXT files are loaded with UTF-8 encoding.

### 3. Text splitting

Documents are split using LangChain's `RecursiveCharacterTextSplitter`.

Current defaults:

```text
chunk_size = 700
chunk_overlap = 100
```

The splitter uses paragraph, newline, sentence, space, and character boundaries as separators.

### 4. Embeddings

Each chunk is converted into a vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model is initialized lazily and reused through the application.

### 5. Vector storage

ChromaDB stores the document chunks persistently in:

```text
data/chroma_db/
```

The collection name is:

```text
document_chunks
```

Each chunk contains metadata including:

```text
document_id
source
page
file_hash
page_count
chunk_id
```

### 6. Retrieval

When a user asks a question, the question is used for semantic similarity search.

The default configuration retrieves:

```text
TOP_K = 6
```

Retrieval can optionally be restricted to a selected document using `document_id`.

The retriever uses Chroma's:

```text
similarity_search_with_relevance_scores()
```

A configurable relevance threshold exists, but the default is `-1.0`, which effectively disables score filtering because score ranges depend on the Chroma distance configuration.

### 7. Context construction

Retrieved chunks are formatted into a context containing:

```text
Source N: filename | Page N
chunk content
```

The context is capped by:

```text
MAX_CONTEXT_CHARS = 12000
```

### 8. LLM generation

The retrieved context and user's question are passed to the Groq-hosted model through LangChain's `ChatOpenAI` integration using Groq's OpenAI-compatible endpoint.

Current default model:

```text
openai/gpt-oss-20b
```

Temperature is set to:

```text
0
```

The prompt instructs the model to answer only from supplied document context, avoid outside knowledge, avoid inventing facts/citations, and use the fixed fallback when the answer is not present:

```text
I could not find this information in the uploaded documents.
```

### 9. Sources

Each returned source contains:

```text
document_id
filename
page
chunk_id
score
```

The frontend groups sources by document and displays relevant pages.

## Duplicate Prevention

Duplicate prevention uses a SHA-256 hash of the complete uploaded file content.

```text
Uploaded file
      │
      ▼
SHA-256(content)
      │
      ▼
Check ChromaDB
      │
 ┌────┴────┐
 │         │
Exists    New
 │         │
 ▼         ▼
Return    Create UUID
existing  Save + index
document
```

This prevents the same file content from being indexed multiple times even if it is uploaded again.

The frontend also checks the returned `document_id` before adding an uploaded document to React state.

## Document-Specific Search

Users can select a document in the frontend.

When selected, the frontend sends its `document_id` with the query. The retriever applies the Chroma metadata filter:

```python
{"document_id": document_id}
```

This supports both global search and search within one selected document.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload and index a document |
| GET | `/documents` | List indexed documents |
| POST | `/query` | Ask a question using RAG |
| DELETE | `/documents/{document_id}` | Delete a document and its indexed chunks |

### Query request

```json
{
  "question": "What is this document about?",
  "document_id": null,
  "top_k": 6
}
```

`document_id` is optional. `top_k` defaults to 6 and is constrained between 1 and 20.

### Query response

```json
{
  "answer": "Generated answer based on the document context.",
  "sources": [
    {
      "document_id": "document-uuid",
      "filename": "example.pdf",
      "page": 2,
      "chunk_id": "document-uuid_1",
      "score": 0.1234
    }
  ]
}
```

## Project Structure

```text
ai-document-intelligence/
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── config.py
│       ├── document_loader.py
│       ├── document_service.py
│       ├── embeddings.py
│       ├── llm.py
│       ├── main.py
│       ├── rag.py
│       ├── retriever.py
│       ├── schemas.py
│       ├── text_splitter.py
│       └── vector_store.py
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Chat.jsx
│       │   ├── DocumentList.jsx
│       │   ├── DocumentUpload.jsx
│       │   ├── Header.jsx
│       │   └── SourceCard.jsx
│       ├── services/
│       │   └── api.js
│       ├── App.jsx
│       ├── main.jsx
│       └── index.css
│
├── data/
│   ├── uploaded_docs/
│   └── chroma_db/
│
├── docker-compose.yml
├── .dockerignore
└── .env
```

## Configuration

Example environment variables:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=700
CHUNK_OVERLAP=100
TOP_K=6
MAX_CONTEXT_CHARS=12000
MIN_RELEVANCE_SCORE=-1.0
```

**Never commit the real `.env` file or API key to GitHub.**

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

### Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Running with Docker

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up
```

Stop:

```bash
docker compose down
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

The Compose setup mounts:

```text
./data:/app/data
```

so uploaded documents and ChromaDB data persist outside the container.

During the current development setup, the React frontend is started separately with:

```bash
cd frontend
npm run dev
```

## Error Handling

### Upload

- PDF/TXT only
- Empty files rejected
- Frontend maximum size: 10 MB
- Backend validates extension and empty content
- Failed indexing removes the partially saved file

### API

The frontend preserves FastAPI error details when available. Query network failures produce a user-friendly backend connection message.

### Delete

The frontend disables the delete button while deletion is in progress and clears the selected document when it is deleted.

## Persistence

```text
data/uploaded_docs/
    Original uploaded files

data/chroma_db/
    ChromaDB vector store
```

ChromaDB uses a persistent directory rather than an in-memory store.

## Technology Stack

**Frontend:** React, JavaScript/JSX, Vite, Tailwind CSS

**Backend:** Python, FastAPI, Uvicorn, Pydantic

**AI/RAG:** LangChain, Hugging Face Sentence Transformers, ChromaDB, Groq, OpenAI-compatible LLM integration

**Deployment:** Docker, Docker Compose, WSL 2 for Windows development

## Design Decisions

### Why RAG?

The application answers questions about user-provided documents. RAG retrieves relevant document chunks and provides them to the LLM as context rather than relying only on general model knowledge.

### Why embeddings?

Embeddings represent text as vectors, allowing semantic similarity search rather than relying only on exact keyword matches.

### Why chunking?

Chunking divides documents into smaller, retrievable sections. Overlap helps preserve context across chunk boundaries.

### Why ChromaDB?

Chroma provides vector storage and similarity search with persistent local storage, fitting the project's indexing workflow.

### Why source metadata?

Source metadata makes generated answers traceable to uploaded documents and pages.

## Current Limitations

- Supported document formats are currently PDF and TXT.
- Relevance-score filtering is disabled by default until retrieval scores are evaluated on a representative dataset.
- The frontend currently runs separately from the Dockerized backend during development.
- No measured retrieval or answer accuracy score is claimed yet.

## Future Improvements

- Build a curated RAG evaluation dataset.
- Measure retrieval recall/precision and answer faithfulness.
- Tune chunk size and overlap using evaluation results.
- Calibrate the relevance-score threshold.
- Add authentication and per-user document isolation.
- Add more document formats such as DOCX.
- Add streaming LLM responses.
- Containerize the production frontend.
- Add automated tests and CI/CD.
- Add observability and structured logging.

## Interview Summary

> **AI Document Intelligence is a full-stack RAG application that lets users upload PDF/TXT documents and ask natural-language questions about them. Documents are loaded, recursively chunked, embedded using a Hugging Face sentence-transformer model, and persisted in ChromaDB. At query time, the system performs semantic Top-K retrieval, optionally filters by document ID, builds a bounded context, and sends that context to a Groq-hosted LLM through an OpenAI-compatible interface. The model is instructed to answer only from the retrieved context, and the API returns source document and page metadata for traceability. The FastAPI backend is Dockerized with persistent data mounted from the host.**
