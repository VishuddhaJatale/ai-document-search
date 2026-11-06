from eval.run_eval import _first_hit_rank


def test_evaluation_prefers_chunk_level_ground_truth():
    item = {"expected_filename": "policy.pdf", "expected_pages": [1], "expected_chunk_ids": ["right"]}
    sources = [
        {"filename": "policy.pdf", "page": 1, "chunk_id": "wrong"},
        {"filename": "policy.pdf", "page": 2, "chunk_id": "right"},
    ]
    assert _first_hit_rank(sources, item) == 2


def test_evaluation_falls_back_to_filename_and_page_labels():
    item = {"expected_filename": "policy.pdf", "expected_pages": [2]}
    sources = [
        {"filename": "policy.pdf", "page": 1, "chunk_id": "c1"},
        {"filename": "policy.pdf", "page": 2, "chunk_id": "c2"},
    ]
    assert _first_hit_rank(sources, item) == 2
