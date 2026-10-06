from __future__ import annotations
import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any, Optional

from src.utils.logging import get_logger
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_seed
from src.inference.generator import SLMQueryRewriter
from src.retrieval.retriever import DenseRetriever
from src.evaluation.retrieval_metrics import (
    evaluate_retrieval_ranking,
    aggregate_metrics,
    compute_relative_improvement,
)
from src.evaluation.diversity import evaluate_query_diversity
from src.evaluation.qualitative import generate_qualitative_report

logger = get_logger("evaluation.benchmark")

def run_retrieval_benchmark(
    benchmark_file: str = "data/benchmarks/retrieval_benchmark.json",
    corpus_file: str = "data/benchmarks/retrieval_corpus.json",
    adapter_path: Optional[str] = None,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str = "experiments",
    num_eval_queries: int = 150,
    k_values: List[int] = [1, 2, 3, 5, 10],
    temperatures: List[float] = [0.0, 0.3, 0.7, 1.0],
    device: str = "cuda",
) -> Dict[str, Any]:
    set_seed(42)
    os.makedirs(resolve_path(output_dir), exist_ok=True)

    with open(resolve_path(benchmark_file), "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    if num_eval_queries and num_eval_queries < len(bench_data):
        random.seed(42)
        eval_queries = random.sample(bench_data, num_eval_queries)
    else:
        eval_queries = bench_data

    logger.info(f"Loaded {len(eval_queries)} benchmark queries for evaluation.")

    # Initialize Retriever
    logger.info("Initializing DenseRetriever...")
    retriever = DenseRetriever(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )

    # Initialize Rewriter
    if not adapter_path:
        candidate_paths = [
            resolve_path("experiments/qlora_qwen/best_adapter"),
            resolve_path("experiments/qlora_qwen/final_adapter"),
            resolve_path("experiments/qlora_qwen/latest_adapter"),
        ]
        adapter_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    logger.info(f"Initializing SLMQueryRewriter with adapter: {adapter_path}...")
    rewriter = SLMQueryRewriter(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        load_in_4bit=True,
        device=device,
    )

    results_table: Dict[str, Dict[str, Any]] = {}
    detailed_predictions: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. Baseline: Original Query
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline: Original Query...")
    baseline_metrics_list = []
    t0 = time.time()
    for item in eval_queries:
        q = item["original_query"]
        docs = retriever.retrieve_single(q, top_k=20)
        retrieved_ids = [d["doc_id"] for d in docs]
        m = evaluate_retrieval_ranking(retrieved_ids, item["relevant_doc_ids"])
        baseline_metrics_list.append(m)
    baseline_lat = round(((time.time() - t0) / len(eval_queries)) * 1000.0, 2)
    baseline_agg = aggregate_metrics(baseline_metrics_list)
    baseline_agg["latency_ms"] = baseline_lat
    results_table["Original Query"] = baseline_agg

    # -------------------------------------------------------------
    # 2. Upper Bound: Gold Human Rewrite
    # -------------------------------------------------------------
    logger.info("Evaluating Gold Human Rewrite...")
    gold_metrics_list = []
    t0 = time.time()
    for item in eval_queries:
        q = item["gold_rewrite"]
        docs = retriever.retrieve_single(q, top_k=20)
        retrieved_ids = [d["doc_id"] for d in docs]
        m = evaluate_retrieval_ranking(retrieved_ids, item["relevant_doc_ids"])
        gold_metrics_list.append(m)
    gold_lat = round(((time.time() - t0) / len(eval_queries)) * 1000.0, 2)
    gold_agg = aggregate_metrics(gold_metrics_list)
    gold_agg["latency_ms"] = gold_lat
    results_table["Gold Human Rewrite"] = gold_agg

    # -------------------------------------------------------------
    # 3. Multi-Query Experiments across K = [1, 2, 3, 5, 10]
    # -------------------------------------------------------------
    diversity_reports = {}

    for k in k_values:
        label = f"SLM Rewrite (K={k})"
        logger.info(f"Evaluating {label} with RRF fusion...")
        metrics_list = []
        gen_latencies = []
        all_diversities = []

        for idx, item in enumerate(eval_queries):
            orig_q = item["original_query"]
            ctx = item.get("context")
            gt_ids = item["relevant_doc_ids"]

            gen_res = rewriter.rewrite(
                query=orig_q,
                context=ctx,
                k=k,
                temperature=0.7 if k > 1 else 0.0,
                top_p=0.9,
            )
            rewritten_qs = gen_res["rewritten_queries"]
            gen_latencies.append(gen_res["latency_ms"])

            # Diversity metrics
            if k > 1:
                div = evaluate_query_diversity(rewritten_qs, embedder=retriever.embedder)
                all_diversities.append(div)

            # Retrieve with RRF
            ret_res = retriever.retrieve_multi(
                rewritten_qs,
                fusion_method="rrf",
                rrf_k=60,
                top_k=20,
            )
            retrieved_ids = [d["doc_id"] for d in ret_res["documents"]]
            m = evaluate_retrieval_ranking(retrieved_ids, gt_ids)
            metrics_list.append(m)

            # Save qualitative record on K=3
            if k == 3:
                # Find ranks
                orig_docs = retriever.retrieve_single(orig_q, top_k=20)
                orig_rank = 999
                for r, d in enumerate(orig_docs, 1):
                    if d["doc_id"] in gt_ids:
                        orig_rank = r
                        break

                rewr_rank = 999
                for r, d in enumerate(ret_res["documents"], 1):
                    if d["doc_id"] in gt_ids:
                        rewr_rank = r
                        break

                detailed_predictions.append({
                    "id": item.get("query_id", idx),
                    "original_query": orig_q,
                    "context": ctx,
                    "gold_rewrite": item["gold_rewrite"],
                    "rewritten_queries": rewritten_qs,
                    "orig_rank": orig_rank,
                    "rewrite_rank": rewr_rank,
                    "relevant_passage": item.get("context", ""),
                })

        agg = aggregate_metrics(metrics_list)
        avg_gen_lat = round(sum(gen_latencies) / len(gen_latencies), 2)
        agg["latency_ms"] = avg_gen_lat
        results_table[label] = agg

        if all_diversities:
            diversity_reports[f"K={k}"] = {
                "distinct_1": round(sum(d["distinct_1"] for d in all_diversities) / len(all_diversities), 4),
                "distinct_2": round(sum(d["distinct_2"] for d in all_diversities) / len(all_diversities), 4),
                "self_bleu": round(sum(d["self_bleu"] for d in all_diversities) / len(all_diversities), 4),
                "pairwise_cosine_distance": round(sum(d["pairwise_cosine_distance"] for d in all_diversities) / len(all_diversities), 4),
            }

    # -------------------------------------------------------------
    # 4. Multi-Query Fusion Ablation (K=3: RRF vs MaxScore vs MeanScore)
    # -------------------------------------------------------------
    fusion_ablation_results = {}
    logger.info("Evaluating Multi-Query Fusion Ablation (RRF vs Max-Score vs Mean-Score)...")
    for method in ["rrf", "max_score", "mean_score"]:
        metrics_list = []
        for item in eval_queries[:50]:
            orig_q = item["original_query"]
            ctx = item.get("context")
            gt_ids = item["relevant_doc_ids"]

            gen_res = rewriter.rewrite(query=orig_q, context=ctx, k=3, temperature=0.7)
            ret_res = retriever.retrieve_multi(gen_res["rewritten_queries"], fusion_method=method, top_k=20)
            retrieved_ids = [d["doc_id"] for d in ret_res["documents"]]
            metrics_list.append(evaluate_retrieval_ranking(retrieved_ids, gt_ids))
        fusion_ablation_results[method] = aggregate_metrics(metrics_list)

    # -------------------------------------------------------------
    # 5. Temperature Ablation (K=3: T in [0.0, 0.3, 0.7, 1.0])
    # -------------------------------------------------------------
    temp_ablation_results = {}
    logger.info("Evaluating Temperature Ablation...")
    for t in temperatures:
        metrics_list = []
        div_list = []
        for item in eval_queries[:50]:
            orig_q = item["original_query"]
            ctx = item.get("context")
            gt_ids = item["relevant_doc_ids"]

            gen_res = rewriter.rewrite(query=orig_q, context=ctx, k=3, temperature=t)
            ret_res = retriever.retrieve_multi(gen_res["rewritten_queries"], fusion_method="rrf", top_k=20)
            retrieved_ids = [d["doc_id"] for d in ret_res["documents"]]
            metrics_list.append(evaluate_retrieval_ranking(retrieved_ids, gt_ids))
            div_list.append(evaluate_query_diversity(gen_res["rewritten_queries"], retriever.embedder))

        agg = aggregate_metrics(metrics_list)
        agg["self_bleu"] = round(sum(d["self_bleu"] for d in div_list) / len(div_list), 4)
        agg["pairwise_dist"] = round(sum(d["pairwise_cosine_distance"] for d in div_list) / len(div_list), 4)
        temp_ablation_results[f"T={t}"] = agg

    # Compute Relative Improvements against Baseline
    base_mrr = results_table["Original Query"]["mrr@10"]
    base_rec10 = results_table["Original Query"]["recall@10"]
    base_ndcg = results_table["Original Query"]["ndcg@10"]

    summary_rows = []
    for method_name, metrics in results_table.items():
        mrr = metrics["mrr@10"]
        rec10 = metrics["recall@10"]
        ndcg = metrics["ndcg@10"]
        rec20 = metrics["recall@20"]
        lat = metrics.get("latency_ms", 0)

        impr_mrr = compute_relative_improvement(mrr, base_mrr)
        impr_rec10 = compute_relative_improvement(rec10, base_rec10)

        summary_rows.append({
            "Method": method_name,
            "Recall@10": rec10,
            "Recall@20": rec20,
            "MRR@10": mrr,
            "nDCG@10": ndcg,
            "MRR_Improvement_%": impr_mrr,
            "Latency_ms": lat,
        })

    # Save benchmark outputs
    benchmark_payload = {
        "summary": summary_rows,
        "results_table": results_table,
        "diversity_analysis": diversity_reports,
        "fusion_ablation": fusion_ablation_results,
        "temperature_ablation": temp_ablation_results,
        "num_eval_queries": len(eval_queries),
    }

    results_file = os.path.join(resolve_path(output_dir), "benchmark_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    # Generate Qualitative Report
    generate_qualitative_report(detailed_predictions, os.path.join(resolve_path(output_dir), "qualitative_report.md"))

    logger.info("Benchmark evaluation completed. Printing summary table:")
    print("\n" + "=" * 80)
    print(f"{'Method':<25} | {'Recall@10':<10} | {'MRR@10':<8} | {'nDCG@10':<8} | {'Δ MRR (%)':<10} | {'Latency':<8}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['Method']:<25} | {row['Recall@10']:<10.4f} | {row['MRR@10']:<8.4f} | {row['nDCG@10']:<8.4f} | {row['MRR_Improvement_%']:<+10.2f}% | {row['Latency_ms']}ms")
    print("=" * 80 + "\n")

    return benchmark_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--num-queries", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    run_retrieval_benchmark(
        adapter_path=args.adapter_path,
        num_eval_queries=args.num_queries,
        output_dir=args.output_dir,
    )
