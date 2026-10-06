from __future__ import annotations
import pytest
from src.evaluation.retrieval_metrics import (
    compute_recall_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_hit_rate_at_k,
    compute_relative_improvement,
    aggregate_metrics,
)
from src.evaluation.diversity import compute_distinct_n, compute_self_bleu

def test_retrieval_metrics_exact_hit():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    ground_truth = {"doc_1"}

    assert compute_recall_at_k(retrieved, ground_truth, 1) == 1.0
    assert compute_mrr_at_k(retrieved, ground_truth, 10) == 1.0
    assert compute_ndcg_at_k(retrieved, ground_truth, 10) == 1.0
    assert compute_hit_rate_at_k(retrieved, ground_truth, 5) == 1.0

def test_retrieval_metrics_second_position():
    retrieved = ["doc_other", "doc_target", "doc_3"]
    ground_truth = {"doc_target"}

    assert compute_recall_at_k(retrieved, ground_truth, 1) == 0.0
    assert compute_recall_at_k(retrieved, ground_truth, 2) == 1.0
    assert compute_mrr_at_k(retrieved, ground_truth, 10) == 0.5
    assert compute_hit_rate_at_k(retrieved, ground_truth, 1) == 0.0
    assert compute_hit_rate_at_k(retrieved, ground_truth, 2) == 1.0

def test_relative_improvement():
    # baseline MRR: 0.20, new MRR: 0.30 -> +50.0%
    impr = compute_relative_improvement(0.30, 0.20)
    assert impr == 50.0

def test_diversity_metrics():
    queries = [
        "what is retrieval augmented generation",
        "how does rag work",
        "retrieval augmented generation architecture explanation"
    ]
    d1 = compute_distinct_n(queries, n=1)
    assert 0.0 < d1 <= 1.0
    sb = compute_self_bleu(queries, n_gram=2)
    assert 0.0 <= sb <= 1.0

def test_aggregate_metrics_disparate_keys():
    m1 = {"recall@1": 1.0, "mrr@10": 0.5}
    m2 = {"recall@1": 0.0, "hit_rate@10": 1.0}  # different keys
    agg = aggregate_metrics([m1, m2])
    assert agg["recall@1"] == 0.5
    assert agg["mrr@10"] == 0.5
    assert agg["hit_rate@10"] == 1.0
    assert aggregate_metrics([]) == {}
