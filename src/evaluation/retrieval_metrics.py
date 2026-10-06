from __future__ import annotations
import math
from typing import List, Dict, Any, Set

def compute_recall_at_k(retrieved_doc_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Proportion of relevant documents retrieved in the top k."""
    if not ground_truth_ids:
        return 0.0
    top_k_ids = set(retrieved_doc_ids[:k])
    hits = len(top_k_ids.intersection(ground_truth_ids))
    return hits / len(ground_truth_ids)

def compute_mrr_at_k(retrieved_doc_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Mean Reciprocal Rank within top k."""
    if not ground_truth_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0

def compute_ndcg_at_k(retrieved_doc_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    if not ground_truth_ids:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG (all relevant docs ranked at top)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(ground_truth_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def compute_hit_rate_at_k(retrieved_doc_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Binary indicator of whether at least one relevant document was in top k."""
    if not ground_truth_ids:
        return 0.0
    top_k_ids = set(retrieved_doc_ids[:k])
    return 1.0 if len(top_k_ids.intersection(ground_truth_ids)) > 0 else 0.0

def evaluate_retrieval_ranking(
    retrieved_doc_ids: List[str],
    ground_truth_ids: List[str],
) -> Dict[str, float]:
    """Compute all standard retrieval metrics for a single query."""
    gt_set = set(ground_truth_ids)
    return {
        "recall@1": compute_recall_at_k(retrieved_doc_ids, gt_set, 1),
        "recall@5": compute_recall_at_k(retrieved_doc_ids, gt_set, 5),
        "recall@10": compute_recall_at_k(retrieved_doc_ids, gt_set, 10),
        "recall@20": compute_recall_at_k(retrieved_doc_ids, gt_set, 20),
        "mrr@10": compute_mrr_at_k(retrieved_doc_ids, gt_set, 10),
        "ndcg@10": compute_ndcg_at_k(retrieved_doc_ids, gt_set, 10),
        "hit_rate@10": compute_hit_rate_at_k(retrieved_doc_ids, gt_set, 10),
    }

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean across all benchmark query metrics."""
    if not metrics_list:
        return {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    agg = {}
    for k in sorted(all_keys):
        vals = [m[k] for m in metrics_list if k in m]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg

def compute_relative_improvement(new_metric: float, baseline_metric: float) -> float:
    """Calculate relative percentage improvement: (New - Baseline) / Baseline * 100."""
    if baseline_metric <= 0:
        return 0.0
    return round(((new_metric - baseline_metric) / baseline_metric) * 100.0, 2)
