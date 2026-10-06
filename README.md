# ReQuery: SLM-Powered Query Rewriting for RAG Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![PEFT QLoRA](https://img.shields.io/badge/PEFT-QLoRA%204--bit-green.svg)](https://github.com/huggingface/peft)
[![VRAM](https://img.shields.io/badge/VRAM-~6GB%20RTX%204050-purple.svg)]()

**ReQuery** is a research and engineering project for fine-tuning a Small Language Model (SLM, <1B parameters) using **4-bit QLoRA** to perform context-grounded query reformulation, query expansion, and multi-query retrieval in Retrieval-Augmented Generation (RAG) pipelines.


```text
User Query (+ Context)
        │
        ▼
Fine-Tuned SLM (Qwen2.5-0.5B QLoRA)
   │  Sampling (temperature=0.7, top_p=0.9, K generations)
   ├── Query 1: "What training strategy was used in the proposed model?"
   ├── Query 2: "How was the proposed model trained?"
   └── Query K: "..."
        │
        ▼
   Dense Retriever (all-MiniLM-L6-v2 + FAISS Index)
        │
        ▼
   Reciprocal Rank Fusion (RRF, k=60) + Deduplication
        │
        ▼
   Top-N Ranked Passages ──> Downstream RAG Generation
```

---

## Key Results Summary

Benchmarked on held-out test data (100 queries, 916 corpus passages) using exact cosine dense retrieval (`all-MiniLM-L6-v2` + FAISS):

| Method | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Relative MRR Gain | Latency (avg/query) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Query** | 0.5200 | 0.6200 | 0.3556 | 0.3937 | *Baseline* | 24.77 ms |
| **Gold Human Rewrite** | 0.7900 | 0.8500 | 0.6193 | 0.6606 | +74.16% | 13.92 ms |
| **SLM Rewrite (K=1)** | 0.7300 | 0.7800 | 0.5563 | 0.5987 | +56.44% | 2,326.09 ms |
| **SLM Rewrite (K=2)** | 0.7500 | 0.8400 | 0.5707 | 0.6140 | +60.49% | 2,649.63 ms |
| **SLM Rewrite (K=3)** | **0.7900** | 0.8100 | 0.5676 | **0.6214** | +59.62% | 2,816.61 ms |
| **SLM Rewrite (K=5)** | 0.7400 | 0.8200 | **0.5778** | 0.6179 | **+62.49%** | 3,182.81 ms |
| **SLM Rewrite (K=10)** | 0.7700 | 0.8100 | 0.5737 | 0.6209 | +61.33% | 3,107.99 ms |

> [!NOTE]
> **Inference Latency & Hardware Note:** SLM generation latency currently averages ~2.3s – 3.1s per query because benchmarks were conducted on a resource-constrained consumer machine (NVIDIA RTX 4050 Laptop GPU, 6GB VRAM, low mobile TGP) performing sequential, unbatched token-by-token 4-bit NF4 autoregressive decoding via vanilla PyTorch. In production environments equipped with datacenter GPUs (e.g., A100 / H100) or high-throughput inference runtimes (such as vLLM, TensorRT-LLM, or llama.cpp), latency for a 0.5B model can be brought down to under 100ms – 150ms.

- **Retrieval Performance Highlights**:
  - SLM-powered reformulation significantly boosts retrieval metrics: **Recall@10 jumps from 0.5200 to 0.7900** (at $K=3$, matching the human gold rewrite benchmark).
  - **MRR@10 improves by up to +62.49%** (at $K=5$), placing relevant documents considerably higher in the ranked list.
  - Multi-query expansion ($K \ge 2$) with Reciprocal Rank Fusion (RRF) consistently outperforms single greedy rewrite generation ($K=1$).
- **Downstream RAG Quality**: Context-to-answer token F1 improves by **+25.12%** compared to the original query baseline.
- **Memory Footprint**: The 4-bit NF4 base model occupies only **~436 MB VRAM**, with training peaking at **4.52 GB VRAM**, operating safely within a consumer 6GB GPU budget.

---

## Project Structure

```text
ReQuery/
│
├── configs/                  # Modular configuration files
│   ├── model.yaml            # Model & generation settings
│   ├── training.yaml         # QLoRA & optimizer hyperparameters
│   ├── retrieval.yaml        # Retriever & fusion parameters
│   └── evaluation.yaml       # Metrics & ablation setups
│
├── data/
│   ├── raw/                  # Downloaded datasets (QReCC, CANARD, AmbigQA)
│   ├── processed/            # Cleaned splits (train, val, test)
│   └── benchmarks/           # Corpus & query evaluation files
│
├── src/
│   ├── data/                 # Download, preprocessing, stats, tokenizer
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── statistics.py
│   │   └── tokenizer.py
│   ├── training/             # 4-bit QLoRA trainer & callbacks
│   │   ├── qlora.py
│   │   ├── callbacks.py
│   │   └── checkpoint.py
│   ├── inference/            # Generator & postprocessor
│   │   ├── generator.py
│   │   └── postprocess.py
│   ├── retrieval/            # Dense retriever, FAISS index, RRF fusion
│   │   ├── retriever.py
│   │   ├── index.py
│   │   └── fusion.py
│   ├── evaluation/           # Metrics, diversity, benchmark, RAG eval
│   │   ├── retrieval_metrics.py
│   │   ├── diversity.py
│   │   ├── qualitative.py
│   │   ├── benchmark.py
│   │   └── rag_eval.py
│   └── utils/                # Config, logging, seed
│       ├── config.py
│       ├── logging.py
│       └── seed.py
│
├── api/                      # FastAPI inference backend
│   └── main.py
├── frontend/                 # Interactive Web UI (HTML5, Vanilla CSS, JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── notebooks/                # Exploratory Jupyter notebooks
│   ├── dataset_analysis.ipynb
│   └── evaluation_analysis.ipynb
├── tests/                    # Pytest test suite (24 passing unit tests)
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_fusion.py
│   ├── test_logging.py
│   ├── test_metrics.py
│   ├── test_postprocess.py
│   ├── test_preprocessing.py
│   └── test_training_callbacks.py
├── experiments/              # Checkpoints, adapter weights, reports, logs
├── requirements.txt
├── run.sh                    # One-click execution script
├── RESEARCH_REPORT.md        # Detailed research report
└── README.md
```

---

## Training Data & Datasets

ReQuery trains the SLM on a curated blend of conversational and open-domain query rewriting datasets:

| Dataset | Focus | Source | Role in ReQuery |
| :--- | :--- | :--- | :--- |
| **QReCC** | Multi-turn conversational QA | `svakulenk0/qrecc` | Teaches resolving dialogue context, historical references, and conversational follow-ups. |
| **CANARD** | Contextual question rewriting | `uva-irlab/canard_quretec` | Specializes in coreference and ellipsis resolution (e.g. converting *"What else did he write?"* into self-contained entity queries). |
| **AmbigQA** | Ambiguous open-domain queries | `sewon/ambig_qa` | Teaches query disambiguation and multi-faceted query expansion. |

### Preprocessing & Data Quality Highlights
- **Strict Leakage Prevention**: Validation and test queries are rigorously excluded from training data to guarantee zero train/val/test data contamination.
- **Cleaning & Deduplication**: Whitespace normalization, artifact stripping, length filtering, and duplicate query pair removal.
- **ChatML Prompt Template**: Formatted with standard system prompts, conversation history (if present), and original query, applying loss masking (`-100`) to prompt tokens so only generated rewrites contribute to gradient updates.
- **Benchmark Corpus**: Automatically extracts paired positive passages and distractors to generate `data/benchmarks/retrieval_corpus.json` and relevance annotations in `retrieval_benchmark.json` for reproducible retrieval benchmarking.

---

## Training Process & Hyperparameters

The SLM fine-tuning pipeline is built around **4-bit QLoRA** designed to run comfortably on consumer-grade hardware (6GB VRAM) while achieving fast convergence and zero data contamination.

### 1. Model Architecture & Quantization
* **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (sub-1B parameters, 502.8M total parameters).
* **Quantization**: 4-bit NormalFloat (`NF4`) with double quantization enabled and `bfloat16` compute dtype.
* **Base VRAM Footprint**: ~436 MB VRAM in 4-bit state, leaving ample memory for activations and optimizer states.

### 2. Parameter-Efficient Fine-Tuning (PEFT / LoRA)
* **LoRA Configuration**: Rank $r = 16$, Scaling Factor $\alpha = 32$, Dropout = $0.05$.
* **Target Modules**: All attention and feed-forward linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Trainable Parameters**: **8,798,208** parameters (**1.75%** of the model), keeping 494.0M base parameters frozen.

### 3. Optimization & Training Dynamics
* **Optimizer**: `paged_adamw_8bit` (automatically offloads optimizer memory spikes to CPU memory if needed).
* **Memory Management**: Gradient checkpointing enabled to discard intermediate activations during forward pass.
* **Effective Batch Size**: 16 (per-device batch size $2 \times$ gradient accumulation steps $8$).
* **Learning Rate & Schedule**: Peak learning rate $2.0 \times 10^{-4}$ with a Cosine annealing schedule and $5\%$ warmup ratio.
* **Loss Masking**: Formatted using the ChatML template; cross-entropy loss is masked with `-100` on system and user prompt tokens so gradient updates strictly optimize rewritten query generation.
* **Hardware & Runtime**:
  * Peak Training VRAM: **4.79 GB** (well within the 6 GB ceiling of mobile RTX 4050).
  * Training Throughput: ~6.91 samples/second (~36 minutes total training runtime for 15,000 processed samples).
  * Final Training Loss: **0.1624**.

### 4. Checkpointing & Model Selection
* **Step-wise Validation**: Evaluated every 50 steps (`eval_steps=50`) on held-out validation queries.
* **Best Adapter Tracking**: The training callback continuously monitors validation loss and saves the optimal model (`best_metric = 0.7575` at step 1,750) to `experiments/qlora_qwen/best_adapter`.
* **Auto-Resume**: Automatically checkpoints every 50 steps and maintains `latest_adapter`, allowing continuous training if epochs are extended without restarting from scratch.

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Activate your conda environment
conda activate ml-env

# Install requirements
pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest tests/
```

### 3. End-to-End One-Click Pipeline

You can run the entire workflow with:

```bash
./run.sh
```

Or execute individual modular steps:

```bash
# Step 1: Download datasets
python -m src.data.download

# Step 2: Preprocess & create leakage-free splits
python -m src.data.preprocess

# Step 3: Compute dataset statistics
python -m src.data.statistics

# Step 4: Fine-tune SLM with 4-bit QLoRA
# Automatically tracks and saves:
#   - Best validation model:  experiments/qlora_qwen/best_adapter
#   - Latest trained model:   experiments/qlora_qwen/latest_adapter
#   - Backward compatibility: experiments/qlora_qwen/final_adapter (best model)
python -m src.training.qlora --epochs 1 --batch-size 4 --grad-accum 2 --max-train-samples 800 --max-val-samples 100

# Continue training from the previous latest model when increasing epochs:
# Simply increase --epochs; it automatically discovers and resumes from the latest checkpoint:
python -m src.training.qlora --epochs 3 --batch-size 4 --grad-accum 2
# (Or pass --no-resume to force a fresh run from scratch)

# Step 5: Run comprehensive retrieval evaluation benchmark (uses best_adapter by default)
python -m src.evaluation.benchmark --num-queries 100

# Step 6: Run downstream RAG answer evaluation (uses best_adapter by default)
python -m src.evaluation.rag_eval
```

---

## Running the API & Web UI

### Start the Server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open your browser at **`http://localhost:8000`** to access the interactive web interface:
- Enter an original query and optional conversation context.
- Adjust $K$ (number of generated queries from 1 to 10) and temperature.
- Choose ranking fusion (RRF, Max-Score, Mean-Score).
- View rewritten queries, individual copy buttons, generation latency, and top retrieved documents with scores.

### API Endpoints:

#### `POST /rewrite`
```bash
curl -X POST http://localhost:8000/rewrite \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did the paper say about training?",
    "context": "Discussion about Attention Is All You Need.",
    "k": 3,
    "temperature": 0.7
  }'
```

#### `POST /rag/search`
```bash
curl -X POST http://localhost:8000/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did the paper say about training?",
    "context": "Discussion about Attention Is All You Need.",
    "k": 3,
    "fusion_method": "rrf",
    "top_k_docs": 5
  }'
```

---

## Citation & References

- **QReCC**: S. Vakulenko et al., *Question Rewriting in Conversational Context*, WSDM 2021.
- **CANARD**: A. Elgohary et al., *Can You Unpack That? Learning to Rewrite Questions for Conversational QA*, EMNLP 2019.
- **AmbigQA**: S. Min et al., *AmbigQA: Answering Ambiguous Open-Domain Questions*, EMNLP 2020.
- **QLoRA**: T. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023.
- **RRF**: G. Cormack et al., *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009.
