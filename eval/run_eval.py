import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
FALLBACK = "I could not find this information in the uploaded documents."


def _post_json(url, payload, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode())


def _retrieve(item, top_k, api, direct, api_key=None):
    if direct:
        from app.rag import retrieve_sources
        return retrieve_sources(item["question"], top_k=top_k).model_dump()
    return _post_json(api.rstrip("/") + "/retrieve", {"question": item["question"], "top_k": top_k}, api_key)


def _answer(item, top_k, api, direct, api_key=None):
    if direct:
        from app.rag import answer_question
        return answer_question(item["question"], top_k=top_k).model_dump()
    return _post_json(api.rstrip("/") + "/query", {"question": item["question"], "top_k": top_k}, api_key)


def _first_hit_rank(sources, item):
    expected_chunks = set(item.get("expected_chunk_ids", []))
    expected_name = item.get("expected_filename", "")
    expected_pages = set(item.get("expected_pages", []))
    for rank, source in enumerate(sources, 1):
        if expected_chunks:
            matched = source.get("chunk_id") in expected_chunks
        else:
            matched = source.get("filename") == expected_name and (not expected_pages or source.get("page") in expected_pages)
        if matched:
            return rank
    return None


def _evaluate(dataset, top_k, api=None, direct=False, skip_answers=False, api_key=None):
    records = []
    hits, reciprocal_ranks, latencies = [], [], []
    keyword_hits = keyword_total = out_scope_total = out_scope_correct = 0
    by_file = {}
    for item in dataset:
        in_scope = item.get("type") == "in_scope"
        retrieval = _retrieve(item, top_k, api, direct, api_key)
        latencies.append(float(retrieval.get("retrieval_latency_seconds") or 0.0))
        rank = _first_hit_rank(retrieval.get("sources", []), item) if in_scope else None
        if in_scope:
            hits.append(1 if rank else 0)
            reciprocal_ranks.append(1 / rank if rank else 0)
            group = by_file.setdefault(item.get("expected_filename") or "unlabelled", {"questions": 0, "hits": 0, "reciprocal_rank_sum": 0.0})
            group["questions"] += 1
            group["hits"] += int(rank is not None)
            group["reciprocal_rank_sum"] += 1 / rank if rank else 0
        record = {"question": item["question"], "type": item.get("type"), "hit_at_k": bool(rank), "rank": rank, "retrieval_latency_seconds": latencies[-1], "retrieved": retrieval.get("sources", [])}
        if not skip_answers:
            result = _answer(item, top_k, api, direct, api_key)
            answer = result.get("answer", "")
            keywords = [word.lower() for word in item.get("expected_answer_keywords", [])]
            keyword_hits += sum(1 for word in keywords if word in answer.lower())
            keyword_total += len(keywords)
            if item.get("type") == "out_of_scope":
                out_scope_total += 1
                out_scope_correct += int(answer.strip() == FALLBACK and not result.get("sources"))
            record.update({"answer": answer, "answer_sources": result.get("sources", [])})
        records.append(record)
    count = len(hits) or 1
    for values in by_file.values():
        questions = values.pop("questions")
        values["hit_at_k"] = values.pop("hits") / questions if questions else None
        values["mrr"] = values.pop("reciprocal_rank_sum") / questions if questions else None
        values["question_count"] = questions
    metrics = {
        f"hit_at_{top_k}": sum(hits) / count,
        "mrr": sum(reciprocal_ranks) / count,
        "mean_retrieval_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "answer_keyword_coverage": keyword_hits / keyword_total if keyword_total else None,
        "out_of_scope_fallback_accuracy": out_scope_correct / out_scope_total if out_scope_total else None,
        "retrieval_by_expected_file": by_file,
        "retrieval_label_type": "chunk_id" if any(item.get("expected_chunk_ids") for item in dataset) else "filename_page",
    }
    return {"metrics": metrics, "records": records}


def _sweep(dataset, top_k, skip_answers=False, api_key=None):
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
            if path.is_file():
                path.unlink()
        text_splitter.CHUNK_SIZE = size
        for filename, content in files:
            document_service.index_document(filename, content)
        reports[str(size)] = _evaluate(dataset, top_k, direct=True, skip_answers=skip_answers, api_key=api_key)
    return reports


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer behavior for AI Document Intelligence")
    parser.add_argument("--dataset", default=str(ROOT / "eval" / "dataset.example.json"))
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None, help="API key for a backend configured with REQUIRE_API_KEY=true; alternatively set API_KEY")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--skip-answers", action="store_true", help="Retrieval-only evaluation: no LLM calls")
    parser.add_argument("--direct", action="store_true", help="Call local Python functions instead of the HTTP API")
    parser.add_argument("--chunk-size-sweep", "--chunk-size", dest="chunk_size_sweep", action="store_true", help="Re-index local files at chunk sizes 500/700/1000 (requires --direct)")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.chunk_size_sweep and not args.direct:
        parser.error("--chunk-size-sweep requires --direct")
    if args.direct:
        sys.path.insert(0, str(ROOT / "backend"))
    api_key = args.api_key or __import__("os").environ.get("API_KEY")
    result = _sweep(dataset, args.top_k, args.skip_answers, api_key) if args.chunk_size_sweep else _evaluate(dataset, args.top_k, api=args.api, direct=args.direct, skip_answers=args.skip_answers, api_key=api_key)
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
        print("| Metric | Result |\n|---|---:|")
        for key, value in result["metrics"].items():
            print(f"| {key} | {json.dumps(value)} |")


if __name__ == "__main__":
    main()
