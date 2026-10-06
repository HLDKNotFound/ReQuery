from __future__ import annotations
import pytest
from src.retrieval.fusion import reciprocal_rank_fusion, max_score_fusion, mean_score_fusion, fuse_rankings

def test_rrf_scoring_and_ranking():
    list_q1 = [
        {"doc_id": "doc_A", "score": 0.9},
        {"doc_id": "doc_B", "score": 0.8},
        {"doc_id": "doc_C", "score": 0.7},
    ]
    list_q2 = [
        {"doc_id": "doc_B", "score": 0.95},
        {"doc_id": "doc_A", "score": 0.85},
        {"doc_id": "doc_D", "score": 0.6},
    ]

    # In list_q1: doc_A is rank 1, doc_B is rank 2
    # In list_q2: doc_B is rank 1, doc_A is rank 2
    # doc_A RRF: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # doc_B RRF: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 (tie)
    fused = reciprocal_rank_fusion([list_q1, list_q2], rrf_k=60, top_n=3)
    assert len(fused) == 3
    assert {fused[0]["doc_id"], fused[1]["doc_id"]} == {"doc_A", "doc_B"}
    assert fused[2]["doc_id"] in ["doc_C", "doc_D"]

def test_max_score_fusion():
    list_q1 = [{"doc_id": "doc_A", "score": 0.5}, {"doc_id": "doc_B", "score": 0.9}]
    list_q2 = [{"doc_id": "doc_A", "score": 0.95}, {"doc_id": "doc_B", "score": 0.2}]
    fused = max_score_fusion([list_q1, list_q2], top_n=2)
    assert fused[0]["doc_id"] == "doc_A"
    assert fused[0]["score"] == 0.95
    assert fused[1]["doc_id"] == "doc_B"
    assert fused[1]["score"] == 0.9

def test_single_query_dispatch():
    list_single = [{"doc_id": "doc_X", "score": 0.88}]
    fused = fuse_rankings([list_single], method="rrf", top_n=1)
    assert len(fused) == 1
    assert fused[0]["doc_id"] == "doc_X"
    assert fused[0]["rank"] == 1

def test_mean_score_fusion():
    list_q1 = [{"doc_id": "doc_A", "score": 0.8}, {"doc_id": "doc_B", "score": 0.6}]
    list_q2 = [{"doc_id": "doc_A", "score": 0.4}]  # doc_B not in q2
    # doc_A appears in both: mean = (0.8 + 0.4) / 2 = 0.6
    # doc_B appears in q1 only: mean = 0.6 / 1 = 0.6
    fused = mean_score_fusion([list_q1, list_q2], top_n=2)
    assert len(fused) == 2
    assert fused[0]["score"] == 0.6
    assert fused[1]["score"] == 0.6
