# AI Document Intelligence frontend

React 19 + Vite + Tailwind CSS interface for the FastAPI RAG backend.

## Development

```bash
npm install --package-lock-only --ignore-scripts
npm ci
cp .env.example .env
npm run dev
```

Set `VITE_API_URL` in `.env` to the backend base URL. It defaults to `http://localhost:8000`.

## Checks

```bash
npm test
npm run build
npm run lint
```

`npm test` runs Vitest and React Testing Library tests. The production container builds the Vite app and serves it with Nginx; see the repository-root README for Docker Compose instructions.
