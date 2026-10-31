import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
FALLBACK = "I could not find this information in the uploaded documents."


def _post_json(url, payload):
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def _evaluate(dataset, top_k, api=None, direct=False):
    if direct:
        from app.rag import answer_question
    records = []
    hits, reciprocal_ranks, keyword_hits, keyword_total = [], [], 0, 0
    out_scope_total = out_scope_correct = 0
    retrieval_times = []
    for item in dataset:
        started = time.perf_counter()
        if direct:
            response = answer_question(item["question"], top_k=top_k)
            result = response.model_dump()
        else:
            result = _post_json(api.rstrip("/") + "/query", {"question": item["question"], "top_k": top_k})
        elapsed = time.perf_counter() - started
        retrieval_times.append(float(result.get("retrieval_latency_seconds") or elapsed))
        sources = result.get("sources", [])
        expected_name = item.get("expected_filename", "")
        expected_pages = set(item.get("expected_pages", []))
        ranks = [i for i, source in enumerate(sources, 1) if source.get("filename") == expected_name and (not expected_pages or source.get("page") in expected_pages)]
        hit = bool(ranks) if item.get("type") == "in_scope" else False
        if item.get("type") == "in_scope":
            hits.append(1 if hit else 0)
            reciprocal_ranks.append(1 / ranks[0] if ranks else 0)
        keywords = [word.lower() for word in item.get("expected_answer_keywords", [])]
        answer = result.get("answer", "").lower()
        keyword_hits += sum(1 for word in keywords if word in answer)
        keyword_total += len(keywords)
        if item.get("type") == "out_of_scope":
            out_scope_total += 1
            out_scope_correct += int(result.get("answer", "").strip() == FALLBACK and not sources)
        records.append({"question": item["question"], "type": item.get("type"), "hit_at_k": bool(hit), "rank": ranks[0] if ranks else None, "retrieval_latency_seconds": retrieval_times[-1], "answer": result.get("answer"), "sources": sources})
    count = len(hits) or 1
    metrics = {
        f"hit_at_{top_k}": sum(hits) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "mean_retrieval_latency_seconds": sum(retrieval_times) / count,
        "answer_keyword_coverage": keyword_hits / keyword_total if keyword_total else None,
        "out_of_scope_fallback_accuracy": out_scope_correct / out_scope_total if out_scope_total else None,
    }
    return {"metrics": metrics, "records": records}


def _sweep(dataset, top_k):
    sys.path.insert(0, str(ROOT / "backend"))
    from app import document_service, text_splitter, vector_store
    upload_dir = document_service.UPLOAD_DIR
    files = []
    for path in upload_dir.iterdir():
        if path.is_file():
            original = path.name.split("_", 1)[1] if "_" in path.name else path.name
            files.append((original, path.read_bytes()))
    if not files:
        raise SystemExit("No indexed upload files found in DATA_DIR/uploaded_docs")
    reports = {}
    for size in (500, 700, 1000):
        shutil.rmtree(vector_store.VECTOR_DIR, ignore_errors=True)
        vector_store.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        vector_store._vector_store = None
        document_service.REGISTRY_PATH.unlink(missing_ok=True)
        for path in upload_dir.iterdir():
            if path.is_file(): path.unlink()
        text_splitter.CHUNK_SIZE = size
        for filename, content in files:
            document_service.index_document(filename, content)
        reports[str(size)] = _evaluate(dataset, top_k, direct=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer behavior for AI Document Intelligence")
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.example.json"))
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--direct", action="store_true", help="Call local Python functions instead of the HTTP API")
    parser.add_argument("--chunk-size-sweep", "--chunk-size", dest="chunk_size_sweep", action="store_true", help="Re-index local files at chunk sizes 500/700/1000 (requires --direct)")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.chunk_size_sweep and not args.direct:
        parser.error("--chunk-size-sweep requires --direct")
    if args.direct:
        sys.path.insert(0, str(ROOT / "backend"))
    result = _sweep(dataset, args.top_k) if args.chunk_size_sweep else _evaluate(dataset, args.top_k, api=args.api, direct=args.direct)
    RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = RESULTS / f"{timestamp}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output.relative_to(ROOT)}")
    if args.chunk_size_sweep:
        print("| Chunk size | Hit@K | MRR | Mean retrieval latency (s) | Keyword coverage | OOS fallback accuracy |")
        print("|---:|---:|---:|---:|---:|---:|")
        for size, report in result.items():
            m = report["metrics"]
            print(f"| {size} | {m[f'hit_at_{args.top_k}']:.3f} | {m['mrr']:.3f} | {m['mean_retrieval_latency_seconds']:.4f} | {m['answer_keyword_coverage']} | {m['out_of_scope_fallback_accuracy']} |")
    else:
        m = result["metrics"]
        print("| Metric | Result |\n|---|---:|")
        for key, value in m.items(): print(f"| {key} | {value} |")

if __name__ == "__main__": main()
