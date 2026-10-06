from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Any
from src.utils.logging import get_logger
from src.utils.config import resolve_path
from src.inference.generator import SLMQueryRewriter
from src.retrieval.retriever import DenseRetriever

logger = get_logger("evaluation.rag_eval")

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth answer."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)

def run_e2e_rag_evaluation(
    benchmark_file: str = "data/benchmarks/retrieval_benchmark.json",
    adapter_path: Optional[str] = None,
    num_samples: int = 30,
) -> Dict[str, Any]:
    """Compare downstream RAG answer quality using original query vs SLM rewritten query."""
    with open(resolve_path(benchmark_file), "r", encoding="utf-8") as f:
        bench_data = json.load(f)[:num_samples]

    candidate_paths = [
        adapter_path,
        resolve_path("experiments/qlora_qwen/best_adapter"),
        resolve_path("experiments/qlora_qwen/final_adapter"),
        resolve_path("experiments/qlora_qwen/latest_adapter"),
    ]
    resolved_adapter = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    retriever = DenseRetriever()
    rewriter = SLMQueryRewriter(adapter_path=resolved_adapter)

    results_orig = []
    results_rewritten = []

    for item in bench_data:
        orig_q = item["original_query"]
        ctx = item.get("context")
        gold_answer = item.get("gold_answer", "") or item.get("gold_rewrite", "")

        # 1. Pipeline with Original Query
        docs_orig = retriever.retrieve_single(orig_q, top_k=3)
        context_text_orig = " ".join(d.get("text", "") for d in docs_orig)
        # Synthesize answer or evaluate context overlap
        f1_orig = compute_token_f1(context_text_orig, gold_answer)
        results_orig.append(f1_orig)

        # 2. Pipeline with SLM Rewritten Query (K=3)
        rewriter_res = rewriter.rewrite(query=orig_q, context=ctx, k=3, temperature=0.7)
        ret_res = retriever.retrieve_multi(rewriter_res["rewritten_queries"], fusion_method="rrf", top_k=3)
        context_text_rewritten = " ".join(d.get("text", "") for d in ret_res["documents"])
        f1_rewritten = compute_token_f1(context_text_rewritten, gold_answer)
        results_rewritten.append(f1_rewritten)

    avg_f1_orig = round(sum(results_orig) / max(len(results_orig), 1), 4)
    avg_f1_rewritten = round(sum(results_rewritten) / max(len(results_rewritten), 1), 4)
    delta = round(((avg_f1_rewritten - avg_f1_orig) / max(avg_f1_orig, 1e-4)) * 100.0, 2)

    report = {
        "num_samples": num_samples,
        "original_query_context_f1": avg_f1_orig,
        "rewritten_query_context_f1": avg_f1_rewritten,
        "relative_improvement_percent": delta,
    }

    out_file = resolve_path("experiments/rag_eval_results.json")
    parent = os.path.dirname(out_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"End-to-end RAG evaluation completed: {report}")
    return report

if __name__ == "__main__":
    run_e2e_rag_evaluation()
