from __future__ import annotations
from typing import List, Dict, Any
from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    rrf_k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Aggregate multiple ranked lists of documents using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum_{q} 1 / (rrf_k + rank(d, q))
    where rank is 1-indexed.
    """
    rrf_scores = defaultdict(float)
    doc_metadata = {}

    for query_results in ranked_lists:
        for rank, item in enumerate(query_results, start=1):
            doc_id = item["doc_id"]
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank)
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = item

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
        item = dict(doc_metadata[doc_id])
        item["score"] = round(score, 6)
        item["rank"] = rank
        item["fusion_method"] = "rrf"
        results.append(item)

    return results

def max_score_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Aggregate rankings by maximum document similarity score across queries."""
    max_scores = defaultdict(float)
    doc_metadata = {}

    for query_results in ranked_lists:
        for item in query_results:
            doc_id = item["doc_id"]
            score = item.get("score", 0.0)
            if score > max_scores[doc_id]:
                max_scores[doc_id] = score
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = item

    sorted_docs = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
        item = dict(doc_metadata[doc_id])
        item["score"] = round(score, 6)
        item["rank"] = rank
        item["fusion_method"] = "max_score"
        results.append(item)

    return results

def mean_score_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Aggregate rankings by mean document similarity score across queries."""
    total_scores = defaultdict(float)
    counts = defaultdict(int)
    doc_metadata = {}

    for query_results in ranked_lists:
        for item in query_results:
            doc_id = item["doc_id"]
            score = item.get("score", 0.0)
            total_scores[doc_id] += score
            counts[doc_id] += 1
            if doc_id not in doc_metadata:
                doc_metadata[doc_id] = item

    mean_scores = {doc_id: total_scores[doc_id] / counts[doc_id] for doc_id in total_scores}
    sorted_docs = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
        item = dict(doc_metadata[doc_id])
        item["score"] = round(score, 6)
        item["rank"] = rank
        item["fusion_method"] = "mean_score"
        results.append(item)

    return results

def fuse_rankings(
    ranked_lists: List[List[Dict[str, Any]]],
    method: str = "rrf",
    rrf_k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Unified ranking fusion dispatcher."""
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        # Single query: return with rank indices
        single = []
        for rank, item in enumerate(ranked_lists[0][:top_n], start=1):
            d = dict(item)
            d["rank"] = rank
            single.append(d)
        return single

    if method == "rrf":
        return reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k, top_n=top_n)
    elif method == "max_score":
        return max_score_fusion(ranked_lists, top_n=top_n)
    elif method == "mean_score":
        return mean_score_fusion(ranked_lists, top_n=top_n)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
