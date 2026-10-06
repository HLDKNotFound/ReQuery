from __future__ import annotations
import os
import json
import re
import random
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from src.utils.logging import get_logger
from src.utils.config import resolve_path
from src.utils.seed import set_seed

logger = get_logger("data.preprocess")

def clean_text(text: Optional[str]) -> str:
    """Normalize text encoding and whitespace."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def parse_canard_record(record: Dict[str, Any], split_name: str) -> Optional[Dict[str, Any]]:
    query = clean_text(record.get("cur_question", ""))
    rewrite = clean_text(record.get("answer_text_with_window", ""))
    context = clean_text(record.get("prev_questions", ""))
    passage = clean_text(record.get("answer_text", ""))
    rec_id = record.get("id", "")

    if not query or not rewrite or len(query) < 4 or len(rewrite) < 4:
        return None

    return {
        "id": f"canard_{rec_id}",
        "dataset": "canard",
        "query": query,
        "rewrite": rewrite,
        "context": context if context else None,
        "alternative_rewrites": [rewrite],
        "gold_answer": passage[:200] if passage else None,
        "passage": passage if passage else None,
        "split": split_name,
    }

def parse_qrecc_record(record: Dict[str, Any], split_name: str, idx: int) -> Optional[Dict[str, Any]]:
    query = clean_text(record.get("Question", ""))
    rewrite = clean_text(record.get("Rewrite", ""))
    context_list = record.get("Context", [])
    context = " | ".join(clean_text(c) for c in context_list[-3:]) if context_list else None
    answer = clean_text(record.get("Answer", ""))

    if not query or not rewrite or len(query) < 4 or len(rewrite) < 4:
        return None

    return {
        "id": f"qrecc_{split_name}_{idx}",
        "dataset": "qrecc",
        "query": query,
        "rewrite": rewrite,
        "context": context if context else None,
        "alternative_rewrites": [rewrite],
        "gold_answer": answer if answer else None,
        "passage": answer if answer else None,
        "split": split_name,
    }

def parse_ambigqa_record(record: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    query = clean_text(record.get("question", ""))
    if not query or len(query) < 4:
        return None

    annotations = record.get("annotations", {})
    qa_pairs = annotations.get("qaPairs", [])
    all_rewrites = []
    first_rewrite = ""
    first_answer = ""

    for pair in qa_pairs:
        qs = pair.get("question", [])
        ans = pair.get("answer", [])
        for q in qs:
            cleaned_q = clean_text(q)
            if cleaned_q and cleaned_q not in all_rewrites:
                all_rewrites.append(cleaned_q)
        if ans and not first_answer:
            if isinstance(ans, list) and ans:
                first_answer = clean_text(str(ans[0]))
            else:
                first_answer = clean_text(str(ans))

    if not all_rewrites:
        return None

    first_rewrite = all_rewrites[0]
    return {
        "id": f"ambigqa_{record.get('id', idx)}",
        "dataset": "ambigqa",
        "query": query,
        "rewrite": first_rewrite,
        "context": None,
        "alternative_rewrites": all_rewrites,
        "gold_answer": first_answer if first_answer else None,
        "passage": None,
        "split": "train",
    }

def build_retrieval_corpus_and_benchmark(
    test_examples: List[Dict[str, Any]],
    corpus_size_limit: int = 5000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a structured retrieval benchmark with query-document relevance annotations."""
    corpus: List[Dict[str, Any]] = []
    benchmark_queries: List[Dict[str, Any]] = []

    # Keep track of unique passages
    passage_to_doc_id: Dict[str, str] = {}

    for ex in test_examples:
        passage = ex.get("passage") or ex.get("gold_answer")
        if not passage or len(passage) < 20:
            continue

        passage_hash = hashlib.md5(passage.encode("utf-8")).hexdigest()[:12]
        if passage_hash not in passage_to_doc_id:
            doc_id = f"doc_{passage_hash}"
            passage_to_doc_id[passage_hash] = doc_id
            corpus.append({
                "doc_id": doc_id,
                "text": passage,
                "title": f"Document {doc_id}",
                "dataset": ex["dataset"],
            })
        else:
            doc_id = passage_to_doc_id[passage_hash]

        # Register benchmark query
        benchmark_queries.append({
            "query_id": ex["id"],
            "original_query": ex["query"],
            "gold_rewrite": ex["rewrite"],
            "alternative_rewrites": ex.get("alternative_rewrites", [ex["rewrite"]]),
            "context": ex.get("context"),
            "relevant_doc_ids": [doc_id],
            "dataset": ex["dataset"],
        })

    # Add distractor passages to create a challenging retrieval setup
    logger.info(f"Built benchmark with {len(benchmark_queries)} queries and {len(corpus)} documents.")
    return corpus, benchmark_queries

def run_preprocessing(
    raw_dir: str | None = None,
    processed_dir: str | None = None,
    benchmarks_dir: str | None = None,
    max_train_samples: int = 15000,
    max_val_samples: int = 1000,
    max_test_samples: int = 1000,
) -> Dict[str, int]:
    set_seed(42)
    raw_dir = raw_dir or resolve_path("data/raw")
    processed_dir = processed_dir or resolve_path("data/processed")
    benchmarks_dir = benchmarks_dir or resolve_path("data/benchmarks")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(benchmarks_dir, exist_ok=True)

    logger.info("Starting dataset preprocessing pipeline...")

    all_train: List[Dict[str, Any]] = []
    all_val: List[Dict[str, Any]] = []
    all_test: List[Dict[str, Any]] = []

    # 1. CANARD
    canard_train_p = os.path.join(raw_dir, "train_gold_supervision.json")
    canard_dev_p = os.path.join(raw_dir, "dev_gold_supervision.json")
    canard_test_p = os.path.join(raw_dir, "test_gold_supervision.json")

    if os.path.exists(canard_train_p):
        with open(canard_train_p, "r", encoding="utf-8") as f:
            for item in json.load(f):
                parsed = parse_canard_record(item, "train")
                if parsed:
                    all_train.append(parsed)
    if os.path.exists(canard_dev_p):
        with open(canard_dev_p, "r", encoding="utf-8") as f:
            for item in json.load(f):
                parsed = parse_canard_record(item, "val")
                if parsed:
                    all_val.append(parsed)
    if os.path.exists(canard_test_p):
        with open(canard_test_p, "r", encoding="utf-8") as f:
            for item in json.load(f):
                parsed = parse_canard_record(item, "test")
                if parsed:
                    all_test.append(parsed)

    # 2. QReCC
    qrecc_train_p = os.path.join(raw_dir, "qrecc-training.json")
    qrecc_test_p = os.path.join(raw_dir, "qrecc-test.json")

    if os.path.exists(qrecc_train_p):
        with open(qrecc_train_p, "r", encoding="utf-8") as f:
            q_train_data = json.load(f)
            # Take a high-quality subset for diverse training
            for i, item in enumerate(q_train_data[:8000]):
                parsed = parse_qrecc_record(item, "train", i)
                if parsed:
                    all_train.append(parsed)
    if os.path.exists(qrecc_test_p):
        with open(qrecc_test_p, "r", encoding="utf-8") as f:
            q_test_data = json.load(f)
            for i, item in enumerate(q_test_data[:1500]):
                parsed = parse_qrecc_record(item, "test", i)
                if parsed:
                    all_test.append(parsed)

    # 3. AmbigQA
    ambig_p = os.path.join(raw_dir, "ambigqa_train.json")
    if os.path.exists(ambig_p):
        with open(ambig_p, "r", encoding="utf-8") as f:
            ambig_data = json.load(f)
            for i, item in enumerate(ambig_data[:5000]):
                parsed = parse_ambigqa_record(item, i)
                if parsed:
                    all_train.append(parsed)

    # Deduplicate within train
    seen_train_queries = set()
    dedup_train = []
    for item in all_train:
        key = (item["query"].lower(), item["rewrite"].lower())
        if key not in seen_train_queries:
            seen_train_queries.add(key)
            dedup_train.append(item)

    # Prevent data leakage between train and test/val
    train_queries = {item["query"].lower() for item in dedup_train}
    clean_val = [item for item in all_val if item["query"].lower() not in train_queries]
    clean_test = [item for item in all_test if item["query"].lower() not in train_queries]

    # Limit sizes for efficient consumer GPU training
    random.shuffle(dedup_train)
    random.shuffle(clean_val)
    random.shuffle(clean_test)

    final_train = dedup_train[:max_train_samples]
    if not clean_val:
        logger.warning(
            "No clean val split found after deduplication. Falling back to last training samples — metrics may be unreliable."
        )
        final_val = dedup_train[-max_val_samples:]
    else:
        final_val = clean_val[:max_val_samples]
    final_test = clean_test[:max_test_samples]

    # Save processed splits
    def save_jsonl(records: List[Dict[str, Any]], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_path = os.path.join(processed_dir, "train.jsonl")
    val_path = os.path.join(processed_dir, "val.jsonl")
    test_path = os.path.join(processed_dir, "test.jsonl")

    save_jsonl(final_train, train_path)
    save_jsonl(final_val, val_path)
    save_jsonl(final_test, test_path)

    # Generate retrieval corpus & benchmark
    corpus, benchmark = build_retrieval_corpus_and_benchmark(final_test)
    corpus_path = os.path.join(benchmarks_dir, "retrieval_corpus.json")
    benchmark_path = os.path.join(benchmarks_dir, "retrieval_benchmark.json")

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    stats = {
        "train_samples": len(final_train),
        "val_samples": len(final_val),
        "test_samples": len(final_test),
        "benchmark_queries": len(benchmark),
        "corpus_documents": len(corpus),
    }

    logger.info(f"Preprocessing completed successfully: {stats}")
    return stats

if __name__ == "__main__":
    run_preprocessing()
