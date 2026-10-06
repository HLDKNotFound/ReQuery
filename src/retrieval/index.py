from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger
from src.utils.config import resolve_path

logger = get_logger("retrieval.index")

def build_faiss_index(
    corpus: List[Dict[str, Any]],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
    batch_size: int = 64,
) -> Tuple[faiss.IndexFlatIP, SentenceTransformer, List[Dict[str, Any]]]:
    """Encode corpus texts and build an IndexFlatIP FAISS index with normalized embeddings."""
    logger.info(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name, device=device)

    texts = [c.get("text", "") for c in corpus]
    logger.info(f"Encoding {len(texts)} corpus passages...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Built FAISS IndexFlatIP with {index.ntotal} vectors of dimension {dim}.")

    return index, model, corpus

def save_index(
    index: faiss.IndexFlatIP,
    corpus: List[Dict[str, Any]],
    index_file: str,
    metadata_file: str,
):
    idx_parent = os.path.dirname(index_file)
    if idx_parent:
        os.makedirs(idx_parent, exist_ok=True)
    meta_parent = os.path.dirname(metadata_file)
    if meta_parent:
        os.makedirs(meta_parent, exist_ok=True)
    faiss.write_index(index, index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved FAISS index to {index_file} and metadata to {metadata_file}.")

def load_index(
    index_file: str,
    metadata_file: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cuda",
) -> Tuple[faiss.IndexFlatIP, SentenceTransformer, List[Dict[str, Any]]]:
    logger.info(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    model = SentenceTransformer(embedding_model_name, device=device)
    logger.info(f"Loaded index with {index.ntotal} entries.")
    return index, model, corpus
