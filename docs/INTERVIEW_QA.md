# AI Document Intelligence — Interview Q&A

These answers describe the current implementation; avoid claiming measured accuracy unless you have run the evaluation on a labeled corpus.

1. **What problem does the project solve?**  It lets users upload documents and ask natural-language questions, returning answers grounded in retrieved passages with document and page references.
2. **What is RAG?**  Retrieval-Augmented Generation first retrieves relevant source text, then supplies that text to an LLM so the answer can be grounded in a private corpus.
3. **Walk through the pipeline.**  FastAPI validates the upload, the loader extracts text, the recursive splitter creates chunks, Hugging Face embeds them, Chroma persists vectors, and queries retrieve context for Groq generation.
4. **Why use embeddings?**  Embeddings represent text as numeric vectors, allowing semantically related questions and passages to be found even when they do not share exact keywords.
5. **Why all-MiniLM-L6-v2?**  It is a compact sentence-transformer model suitable for local semantic embeddings; its actual quality still needs evaluation on the target corpus.
6. **Why ChromaDB?**  It provides a straightforward persistent vector store and metadata filtering for a local/single-service application.
7. **Why cosine distance?**  The collection is configured with `hnsw:space=cosine`; normalized relevance scores make threshold interpretation more consistent than mixing metrics. Old collections need re-indexing.
8. **How did you choose chunk size?**  The default is 700 characters with 100 overlap as a starting point. The evaluation script supports comparing 500, 700, and 1000 rather than asserting one is universally best.
9. **What is Top-K?**  It is the number of candidate chunks retrieved for a question. A larger K can improve coverage but may add irrelevant context and increase prompt size/latency.
10. **How do you prevent duplicate uploads?**  The service hashes file bytes with SHA-256 and checks the registry/vector metadata before indexing. A process-wide lock serializes the check-and-index sequence within one backend process.
11. **How do you handle follow-up questions?**  Up to six prior chat messages can be sent; when history exists, the LLM rewrites the latest question into a standalone query before retrieval.
12. **How does streaming work?**  `/query/stream` emits Server-Sent Events containing token chunks and a final sources event. The frontend falls back to `/query` if streaming fails.
13. **How do you reduce prompt-injection risk?**  The system prompt explicitly labels document context as untrusted data and instructs the model to ignore instructions embedded in that context. This is a mitigation, not a formal security boundary.
14. **What are the current limitations?**  There is no OCR for scanned PDFs, provider inference needs Groq/network access, hybrid BM25 search is not implemented, and the JSON registry lock is process-local.
15. **What would you change for 10,000+ documents or production scale?**  Benchmark chunk counts and latency, move ingestion to background workers, use a transactional registry/database and distributed locking, add observability and backups, and evaluate a managed/distributed vector store if the workload requires it.
