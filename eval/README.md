# RAG evaluation

- `dataset.json` is the labelled evaluation set: 33 in-scope and 6 out-of-scope questions over three documents (`15._KRR-CD-Theory.pdf`, `week_6_Logic-Programming.pdf`, `New_Text_Document.txt`). Upload those same files before running the evaluation.
- `dataset.example.json` is a placeholder showing the format. To evaluate your own documents, copy it, replace every entry with questions, exact filenames, page numbers and answer keywords from your corpus, and pass it with `--dataset`.

Each entry has `question`, `type` (`in_scope` or `out_of_scope`), `expected_filename`, `expected_pages`, and `expected_answer_keywords`.

## API evaluation

1. Start the backend and upload the documents referenced by the dataset.
2. Run from the repository root:

```bash
# Retrieval only: no LLM calls, fast and free
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6 --skip-answers

# Retrieval + answers (uses your Groq quota)
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6
```

The script reports:

- **Hit@K** and **MRR**, measured through the `/retrieve` endpoint, so they measure the retriever only. Only in-scope questions count.
- **Mean retrieval latency**, from the backend's retrieval timing field.
- **Answer keyword coverage**, a substring match of expected keywords in the answer (a crude check: "termination" does not match "terminate").
- **Out-of-scope fallback accuracy**, the share of out-of-scope questions that got the fallback answer with no sources.

It saves a timestamped JSON report under `eval/results/` and prints a Markdown summary table. Runs started within the same second overwrite each other's file, so rename a report you want to keep (for example `baseline.json`).

## Known limits of this evaluation

- Hit@K uses page labels. For TXT files every chunk has page 1, so TXT hits are not meaningful and should be excluded from reported retrieval numbers.
- Page labels are strict: a correct answer found on an adjacent slide counts as a miss.
- With 33 in-scope questions, one question changes Hit@K by about 3 points.

## Chunk-size sweep

A sweep re-indexes the files already in `DATA_DIR/uploaded_docs` at chunk sizes 500, 700 and 1000. It uses the local Python pipeline, so the embedding model must be available locally and a valid Groq key must be configured for answer generation:

```bash
DATA_DIR=./data GROQ_API_KEY=... python eval/run_eval.py --direct --chunk-size-sweep --dataset eval/dataset.json --top-k 6
```

The sweep deletes and rebuilds the local Chroma collection and document registry. Back up `data/` first. It is intentionally not an API operation, because the running service's chunk configuration is fixed at startup. Add `--skip-answers` to run the sweep without LLM calls.
