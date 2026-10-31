# Changelog

- `.github/workflows/ci.yml` — added backend pytest and frontend install/test/build CI jobs.
- `.gitignore` — excludes secrets, Python/Node build artifacts, runtime data, and generated eval reports.
- `README.md` — documented updated architecture, endpoints, tests, evaluation, Docker, trade-offs, and limitations.
- `docker-compose.yml` — added `/app/data`, CORS for dev/production origins, backend healthcheck, frontend service, and persistent Hugging Face cache.
- `backend/.dockerignore` — keeps test/dev artifacts and local data out of the backend image.
- `backend/.env.example` — documented safe placeholder configuration, upload limit, CORS, and MMR.
- `backend/Dockerfile` — clarified the `/app` layout and runtime setup.
- `backend/requirements.txt` — pinned compatible runtime dependencies and added explicit text-splitter dependency.
- `backend/requirements-dev.txt` — added a lighter test dependency set without sentence-transformers/PyTorch.
- `backend/app/config.py` — added configurable data path, upload limit, CORS, MMR, and cosine-score threshold defaults.
- `backend/app/document_loader.py` — added non-UTF-8 TXT decoding and robust PDF extraction fallback/error handling.
- `backend/app/document_service.py` — added registry persistence, chunk counts, SHA-256 deduplication, upload/delete locking, rollback handling, and stale-registry recovery after a vector-store rebuild.
- `backend/app/embeddings.py` — made Hugging Face embeddings lazy so tests can inject fakes without importing the model stack.
- `backend/app/main.py` — added streamed upload limits, file validation, threadpool indexing, SSE query route, and expanded health response.
- `backend/app/rag.py` — fixed page numbering, centralized fallback, added history rewriting, prompt-injection hardening, streaming, and timing logs.
- `backend/app/retriever.py` — added optional MMR and cosine relevance filtering.
- `backend/app/schemas.py` — added validated chat history and optional/config-driven Top-K.
- `backend/app/vector_store.py` — configured cosine distance and added vector-store health/count helpers.
- `backend/tests/conftest.py` — configured isolated test storage, deterministic fake embeddings, and fake LLM.
- `backend/tests/test_unit.py` — added filename/hash/splitting/context/page/schema tests.
- `backend/tests/test_api.py` — added API upload/query/health/stream/delete coverage.
- `frontend/.dockerignore` — excludes local build/dependency/environment files from production builds.
- `frontend/.env.example` — documented `VITE_API_URL`.
- `frontend/.gitignore` — excludes local environment files.
- `frontend/Dockerfile` — added multi-stage Vite build and Nginx production serving.
- `frontend/nginx.conf` — added SPA history fallback routing.
- `frontend/README.md` — replaced template instructions with project-specific development/test commands.
- `frontend/eslint.config.js` — added Vitest globals for test files.
- `frontend/package.json` — added React Markdown, Vitest, jsdom, and React Testing Library dependencies/scripts.
- `frontend/src/App.jsx` — replaced delete alerts with inline errors.
- `frontend/src/components/Chat.jsx` — added history, streaming UI, Markdown answers, and clear-chat control.
- `frontend/src/components/DocumentList.jsx` — displayed document page and chunk counts.
- `frontend/src/components/DocumentUpload.jsx` — clarified 10 MB client-side validation message.
- `frontend/src/index.css` — added dark-theme Markdown styles.
- `frontend/src/services/api.js` — added environment-based API URL, history payload, SSE parsing, and JSON fallback.
- `frontend/src/test-setup.js` — configured Testing Library matchers.
- `frontend/src/__tests__/DocumentUpload.test.jsx` — added upload validation tests.
- `frontend/src/__tests__/SourceCard.test.jsx` — added source grouping test.
- `frontend/vite.config.js` — configured Vitest/jsdom.
- `eval/README.md` — documented dataset preparation, metrics, and evaluation commands.
- `eval/dataset.example.json` — added six clearly identified placeholder evaluation cases.
- `eval/run_eval.py` — added API/direct evaluation, retrieval/answer metrics, timestamped JSON output, and chunk-size sweep.
- `eval/results/.gitkeep` — retained the output directory in source control.
- `docs/INTERVIEW_QA.md` — added 15 code-grounded interview questions and answers.

## Verification caveat

Python compilation and static JSON/YAML/JavaScript syntax checks passed. Full pytest, Vitest, frontend build, and Docker execution could not be completed in this environment: Python package installation was blocked by DNS/network failure, Node dependencies were unavailable, and Docker was not installed. `frontend/package-lock.json` could not be regenerated offline; CI and the frontend Docker build refresh it with `npm install --package-lock-only --ignore-scripts` before `npm ci`.
