#!/bin/bash
set -e

echo "=========================================================="
echo "  ReQuery: SLM RAG Query Rewriter End-to-End Pipeline     "
echo "=========================================================="

# 1. Activate conda environment
# conda activate ml-env

# 2. Download Datasets
echo "[1/6] Downloading Datasets (QReCC, CANARD, AmbigQA)..."
python -m src.data.download

# 3. Preprocess and Split Datasets
echo "[2/6] Preprocessing and Generating Benchmark Splits..."
python -m src.data.preprocess

# 4. Generate Dataset Statistics
echo "[3/6] Computing Dataset Statistics..."
python -m src.data.statistics

# 5. Fine-Tune SLM with 4-bit QLoRA
echo "[4/6] Fine-Tuning Qwen2.5-0.5B with 4-bit QLoRA..."
python -m src.training.qlora --epochs 1 --batch-size 4 --grad-accum 2 --max-train-samples 800 --max-val-samples 100

# 6. Run Benchmark Evaluation Suite (evaluates best validation adapter)
echo "[5/6] Running Retrieval Benchmark Suite..."
python -m src.evaluation.benchmark --num-queries 100

# 7. Run Downstream RAG Evaluation
echo "[6/6] Running Downstream End-to-End RAG Evaluation..."
python -m src.evaluation.rag_eval

echo "=========================================================="
echo "Pipeline finished! To launch the API and Web UI, run:"
echo "uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "=========================================================="
