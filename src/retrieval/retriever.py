from __future__ import annotations
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger
from src.utils.config import load_yaml, resolve_path
from src.retrieval.index import build_faiss_index, load_index, save_index
from src.retrieval.fusion import fuse_rankings

logger = get_logger("retrieval.retriever")

class DenseRetriever:
    """Vector database retrieval engine with multi-query fusion support."""

    def __init__(
        self,
        index_file: Optional[str] = None,
        metadata_file: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.embedding_model_name = embedding_model_name

        index_file = index_file or resolve_path("data/benchmarks/faiss_index.bin")
        metadata_file = metadata_file or resolve_path("data/benchmarks/faiss_metadata.json")

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self.index, self.embedder, self.corpus = load_index(
                index_file, metadata_file, embedding_model_name, self.device
            )
        else:
            corpus_path = resolve_path("data/benchmarks/retrieval_corpus.json")
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(f"Corpus file not found at {corpus_path}. Please run preprocessing first.")
            import json
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            self.index, self.embedder, self.corpus = build_faiss_index(
                corpus, embedding_model_name, self.device
            )
            save_index(self.index, self.corpus, index_file, metadata_file)

    def retrieve_single(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve top_k documents for a single query."""
        emb = self.embedder.encode([query], normalize_embeddings=True)
        emb = np.ascontiguousarray(emb, dtype=np.float32)

        scores, indices = self.index.search(emb, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.corpus):
                continue
            doc = dict(self.corpus[idx])
            doc["score"] = round(float(score), 4)
            doc["rank"] = rank
            doc["matched_query"] = query
            results.append(doc)
        return results

    def retrieve_multi(
        self,
        queries: List[str],
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Retrieve documents for multiple queries and fuse rankings using RRF or score aggregation."""
        start_time = time.time()
        if not queries:
            return {"documents": [], "latency_ms": 0.0, "queries_used": []}

        valid_queries = [q.strip() for q in queries if q.strip()]
        if not valid_queries:
            return {"documents": [], "latency_ms": 0.0, "queries_used": []}

        embs = self.embedder.encode(valid_queries, normalize_embeddings=True, show_progress_bar=False)
        embs = np.ascontiguousarray(embs, dtype=np.float32)
        scores_mat, indices_mat = self.index.search(embs, min(top_k, self.index.ntotal))

        ranked_lists = []
        for q_idx, q in enumerate(valid_queries):
            query_results = []
            for rank, (score, idx) in enumerate(zip(scores_mat[q_idx], indices_mat[q_idx]), start=1):
                if idx < 0 or idx >= len(self.corpus):
                    continue
                doc = dict(self.corpus[idx])
                doc["score"] = round(float(score), 4)
                doc["rank"] = rank
                doc["matched_query"] = q
                query_results.append(doc)
            ranked_lists.append(query_results)

        # Fuse rankings across all query outputs
        fused_documents = fuse_rankings(
            ranked_lists,
            method=fusion_method,
            rrf_k=rrf_k,
            top_n=top_k,
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "documents": fused_documents,
            "queries_used": valid_queries,
            "fusion_method": fusion_method,
            "latency_ms": round(elapsed_ms, 2),
        }

def get_default_retriever() -> DenseRetriever:
    cfg = load_yaml(resolve_path("configs/retrieval.yaml")).get("retrieval", {})
    return DenseRetriever(
        embedding_model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=cfg.get("device", "cuda"),
    )
