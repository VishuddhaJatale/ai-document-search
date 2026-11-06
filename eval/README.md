# RAG evaluation

The labelled set in `dataset.json` currently contains 33 in-scope and 6 out-of-scope questions over three source documents. Upload those exact files before evaluating. The set is a starting benchmark, not a statistically representative test corpus.

Each in-scope row should identify the relevant source using either:
- `expected_chunk_ids`: preferred, precise chunk-level ground truth; or
- `expected_filename` plus `expected_pages`: legacy fallback. Page labels are weak supervision, especially for TXT files where every chunk has page 1.

For a useful evaluation, grow the dataset with questions from each document, including exact-term/acronym queries, paraphrases, multi-passage questions, ambiguous questions and genuinely out-of-scope questions. Keep a held-out set that is not used while tuning retrieval. Do not invent expected chunk IDs: derive them from the indexed corpus and inspect the text.

## API evaluation

1. Start the backend and upload all files used by the dataset.
2. Run from the repository root:

```bash
# Retrieval only: no LLM calls
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6 --skip-answers

# Retrieval + answer behavior (uses Groq quota)
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6
```

If API key protection is enabled, provide `--api-key YOUR_KEY` or set `API_KEY` in the shell environment. Avoid putting secrets directly in shell history or committed files.

## Metrics and interpretation

- **Hit@K**: share of in-scope questions with at least one labelled chunk/page in the top K.
- **MRR**: average reciprocal rank of the first labelled result.
- **Mean retrieval latency**: latency returned by the retrieval endpoint.
- **Answer keyword coverage**: a lightweight substring heuristic only; it is not a correctness or faithfulness score.
- **Out-of-scope fallback accuracy**: share of out-of-scope questions that return the exact fallback answer and no sources.
- **Per-file metrics**: Hit@K and MRR are also reported per expected filename.

Chunk IDs are the recommended ground truth. If the dataset uses only filename/page labels, TXT results are not reliable and adjacent-page results can be false negatives. The report records which label type was used. A question with multiple relevant passages can list multiple chunk IDs; the current Hit@K/MRR use first relevant rank, while Hit@K does not require every relevant chunk.

For answer quality, manually review a stratified sample for factual correctness, faithfulness to retrieved passages, citation support and refusal behavior. Do not present keyword coverage as answer faithfulness. Add human-reviewed labels before reporting a faithfulness percentage.

## Comparing dense and hybrid retrieval

Run the same dataset twice, changing only `RETRIEVAL_MODE`:

```bash
# In backend/.env
RETRIEVAL_MODE=dense
```

Restart backend and run retrieval-only evaluation. Then set `RETRIEVAL_MODE=hybrid`, restart and run again. Keep Top-K, corpus, chunking, embedding model and dataset fixed. Compare Hit@K, MRR, per-file metrics and latency. Save the timestamped JSON reports under `eval/results/` and record the configuration next to each report.

Hybrid mode combines dense Chroma results with BM25 lexical results using reciprocal rank fusion. The BM25 candidate corpus is read from the same Chroma collection, so document filters and deletions stay aligned with the vector index. Dense mode remains available for baseline comparisons.

## Chunk-size sweep

The optional sweep re-indexes local files at chunk sizes 500, 700 and 1000. It deletes and rebuilds the local Chroma collection and registry. Back up `data/` first:

```bash
DATA_DIR=./data GROQ_API_KEY=... python eval/run_eval.py --direct --chunk-size-sweep --dataset eval/dataset.json --top-k 6 --skip-answers
```

Do not run this against your only copy of production data.
