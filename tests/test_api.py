from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data

def test_rewrite_validation_max_length(client):
    # Query exceeding max_length of 1000 characters should trigger validation error (422)
    oversized_query = "a" * 1001
    response = client.post("/rewrite", json={"query": oversized_query})
    assert response.status_code == 422

def test_rewrite_endpoint_mocked(client):
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = {
        "original_query": "What is RAG?",
        "context": None,
        "rewritten_queries": ["What is retrieval augmented generation?"],
        "k_requested": 1,
        "k_generated": 1,
        "latency_ms": 12.34,
        "device": "cpu",
    }

    with patch("api.main.get_rewriter", return_value=mock_rewriter):
        response = client.post("/rewrite", json={"query": "What is RAG?", "k": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["original_query"] == "What is RAG?"
        assert data["rewritten_queries"] == ["What is retrieval augmented generation?"]
        assert data["latency_ms"] == 12.34

def test_rag_search_endpoint_mocked(client):
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = {
        "rewritten_queries": ["What is retrieval augmented generation?"],
        "latency_ms": 15.0,
    }
    mock_retriever = MagicMock()
    mock_retriever.retrieve_multi.return_value = {
        "documents": [{"doc_id": "doc_1", "score": 0.95, "title": "RAG"}],
        "latency_ms": 5.0,
    }

    with patch("api.main.get_rewriter", return_value=mock_rewriter), \
         patch("api.main.get_retriever", return_value=mock_retriever):
        response = client.post("/rag/search", json={"query": "What is RAG?", "k": 1})
        assert response.status_code == 200
        data = response.json()
        assert data["original_query"] == "What is RAG?"
        assert len(data["retrieved_documents"]) == 1
        assert data["fusion_method"] == "rrf"
