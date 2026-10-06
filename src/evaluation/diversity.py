from __future__ import annotations
import math
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

def compute_distinct_n(queries: List[str], n: int = 1) -> float:
    """Compute Distinct-N: number of unique n-grams divided by total n-grams."""
    total_ngrams = 0
    unique_ngrams = set()

    for q in queries:
        tokens = q.strip().lower().split()
        if len(tokens) < n:
            continue
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)

    if total_ngrams == 0:
        return 0.0
    return round(len(unique_ngrams) / total_ngrams, 4)

def compute_pairwise_cosine_distance(queries: List[str], embedder) -> float:
    """Compute average pairwise cosine distance (1 - cosine similarity) among generated queries."""
    if len(queries) <= 1:
        return 0.0

    embeddings = embedder.encode(queries, normalize_embeddings=True)
    sim_matrix = np.dot(embeddings, embeddings.T)

    n = len(queries)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = float(sim_matrix[i, j])
            # Distance in [0, 2]
            distances.append(max(0.0, 1.0 - cos_sim))

    if not distances:
        return 0.0
    return round(float(np.mean(distances)), 4)

def compute_self_bleu(queries: List[str], n_gram: int = 2) -> float:
    """Compute Self-BLEU: average BLEU score of each generated query using the others as references."""
    if len(queries) <= 1:
        return 1.0

    tokenized_queries = [q.lower().split() for q in queries]
    bleu_scores = []

    for i, hypothesis in enumerate(tokenized_queries):
        references = [tokenized_queries[j] for j in range(len(tokenized_queries)) if j != i]
        if not hypothesis:
            continue

        # Simple n-gram precision
        precisions = []
        hyp_len = len(hypothesis)
        for n in range(1, n_gram + 1):
            if hyp_len < n:
                precisions.append(0.0)
                continue
            hyp_ngrams = Counter(tuple(hypothesis[k:k + n]) for k in range(hyp_len - n + 1))
            max_ref_counts = Counter()
            for ref in references:
                ref_ngrams = Counter(tuple(ref[k:k + n]) for k in range(len(ref) - n + 1))
                for ng in ref_ngrams:
                    max_ref_counts[ng] = max(max_ref_counts[ng], ref_ngrams[ng])

            clipped = {ng: min(count, max_ref_counts[ng]) for ng, count in hyp_ngrams.items()}
            total_clipped = sum(clipped.values())
            total_possible = sum(hyp_ngrams.values())
            precisions.append(total_clipped / total_possible if total_possible > 0 else 0.0)

        # Geometric mean
        if all(p > 0 for p in precisions):
            log_prec = sum(math.log(p) for p in precisions) / len(precisions)
            bleu = math.exp(log_prec)
        else:
            bleu = 0.0
        bleu_scores.append(bleu)

    return round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0

def evaluate_query_diversity(queries: List[str], embedder=None) -> Dict[str, float]:
    """Compute complete diversity report for a candidate set of queries."""
    d1 = compute_distinct_n(queries, n=1)
    d2 = compute_distinct_n(queries, n=2)
    self_bleu = compute_self_bleu(queries, n_gram=2)
    cosine_dist = compute_pairwise_cosine_distance(queries, embedder) if embedder else 0.0

    return {
        "distinct_1": d1,
        "distinct_2": d2,
        "self_bleu": self_bleu,
        "pairwise_cosine_distance": cosine_dist,
    }
