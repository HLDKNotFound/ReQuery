from __future__ import annotations
import os
import time
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.utils.logging import get_logger
from src.utils.config import resolve_path
from src.inference.generator import SLMQueryRewriter
from src.retrieval.retriever import DenseRetriever

logger = get_logger("api.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing ReQuery API service...")
    yield
    logger.info("Shutting down ReQuery API service...")

app = FastAPI(
    title="SLM RAG Query Rewriter API",
    description="Lightweight SLM-powered Query Rewriting and Multi-Query Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model and retriever handles (thread-safe lazy loaded)
rewriter: Optional[SLMQueryRewriter] = None
retriever: Optional[DenseRetriever] = None
_rewriter_lock = threading.Lock()
_retriever_lock = threading.Lock()

def get_rewriter() -> SLMQueryRewriter:
    global rewriter
    if rewriter is None:
        with _rewriter_lock:
            if rewriter is None:
                candidate_paths = [
                    resolve_path("experiments/qlora_qwen/best_adapter"),
                    resolve_path("experiments/qlora_qwen/final_adapter"),
                    resolve_path("experiments/qlora_qwen/latest_adapter"),
                ]
                adapter_path = next((p for p in candidate_paths if os.path.exists(p)), None)
                rewriter = SLMQueryRewriter(
                    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
                    adapter_path=adapter_path,
                    load_in_4bit=True,
                )
    return rewriter

def get_retriever() -> DenseRetriever:
    global retriever
    if retriever is None:
        with _retriever_lock:
            if retriever is None:
                retriever = DenseRetriever(
                    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                )
    return retriever

class RewriteRequest(BaseModel):
    query: str = Field(..., max_length=1000, json_schema_extra={"example": "What did the paper say about training?"})
    context: Optional[str] = Field(None, max_length=2000, json_schema_extra={"example": "Discussion about the Attention Is All You Need paper."})
    k: int = Field(5, ge=1, le=10, description="Number of rewritten queries to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(64, ge=16, le=256)

class RewriteResponse(BaseModel):
    original_query: str
    context: Optional[str]
    rewritten_queries: List[str]
    k_requested: int
    k_generated: int
    latency_ms: float
    device: str

class RAGSearchRequest(BaseModel):
    query: str = Field(..., max_length=1000, json_schema_extra={"example": "What did the paper say about training?"})
    context: Optional[str] = Field(None, max_length=2000)
    k: int = Field(3, ge=1, le=10)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k_docs: int = Field(5, ge=1, le=20)
    fusion_method: str = Field("rrf", description="rrf, max_score, or mean_score")

class RAGSearchResponse(BaseModel):
    original_query: str
    rewritten_queries: List[str]
    retrieved_documents: List[Dict[str, Any]]
    fusion_method: str
    rewriter_latency_ms: float
    retrieval_latency_ms: float
    total_latency_ms: float

@app.get("/health")
def health_check():
    candidate_paths = [
        resolve_path("experiments/qlora_qwen/best_adapter"),
        resolve_path("experiments/qlora_qwen/final_adapter"),
        resolve_path("experiments/qlora_qwen/latest_adapter"),
    ]
    active_adapter = next((p for p in candidate_paths if os.path.exists(p)), None)
    return {
        "status": "healthy",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_adapter_loaded": active_adapter is not None,
        "adapter_path": active_adapter,
    }

@app.post("/rewrite", response_model=RewriteResponse)
def rewrite_query(req: RewriteRequest):
    try:
        model_instance = get_rewriter()
        res = model_instance.rewrite(
            query=req.query,
            context=req.context,
            k=req.k,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
        )
        return RewriteResponse(**res)
    except Exception as e:
        logger.error(f"Error during query rewriting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/search", response_model=RAGSearchResponse)
def rag_search(req: RAGSearchRequest):
    try:
        t0 = time.time()
        model_instance = get_rewriter()
        rewriter_res = model_instance.rewrite(
            query=req.query,
            context=req.context,
            k=req.k,
            temperature=req.temperature,
        )
        t_rewriter = rewriter_res["latency_ms"]

        retriever_instance = get_retriever()
        t1 = time.time()
        ret_res = retriever_instance.retrieve_multi(
            queries=rewriter_res["rewritten_queries"],
            fusion_method=req.fusion_method,
            top_k=req.top_k_docs,
        )
        t_retriever = ret_res["latency_ms"]
        total_time = round((time.time() - t0) * 1000.0, 2)

        return RAGSearchResponse(
            original_query=req.query,
            rewritten_queries=rewriter_res["rewritten_queries"],
            retrieved_documents=ret_res["documents"],
            fusion_method=req.fusion_method,
            rewriter_latency_ms=t_rewriter,
            retrieval_latency_ms=t_retriever,
            total_latency_ms=total_time,
        )
    except Exception as e:
        logger.error(f"Error during RAG search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

frontend_dir = resolve_path("frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "ReQuery API running. Create frontend/index.html to view UI."}

