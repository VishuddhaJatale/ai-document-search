# RAG evaluation

`dataset.example.json` is a **placeholder dataset only**. Replace every example with questions, exact filenames, page numbers, and answer keywords drawn from your own uploaded documents before interpreting the metrics. No evaluation results are claimed by this repository.

## API evaluation

1. Start the backend and upload/index the documents referenced by your dataset.
2. Copy `dataset.example.json` to `dataset.json` and replace the placeholders.
3. Run from the repository root:

```bash
python eval/run_eval.py --dataset eval/dataset.json --api http://localhost:8000 --top-k 6
```

The script reports Hit@K, MRR, mean retrieval latency (from the backend's retrieval timing field), answer keyword coverage, and out-of-scope fallback accuracy. It saves a timestamped JSON report under `eval/results/` and prints a Markdown summary table.

## Chunk-size sweep

A sweep re-indexes the files already present in `DATA_DIR/uploaded_docs` at chunk sizes 500, 700, and 1000. It uses the local Python pipeline, so the embedding model must be available locally and a valid Groq key must be configured for answer generation:

```bash
DATA_DIR=./data GROQ_API_KEY=... python eval/run_eval.py --direct --chunk-size-sweep --dataset eval/dataset.json --top-k 6
```

The sweep deletes and rebuilds the local Chroma collection and document registry. Back up `data/` first. It is intentionally not an API operation because the running service's chunk configuration is fixed at process startup.
