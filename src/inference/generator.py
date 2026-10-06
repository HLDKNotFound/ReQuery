from __future__ import annotations
import os
import time
from typing import List, Dict, Any, Optional
import torch

from src.utils.logging import get_logger
from src.utils.config import load_yaml, resolve_path
from src.data.tokenizer import format_prompt
from src.training.checkpoint import load_rewriter_model
from src.inference.postprocess import clean_query_text, deduplicate_queries

logger = get_logger("inference.generator")

class SLMQueryRewriter:
    """Inference generator for SLM-based query rewriting and multi-query expansion."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: str = "cuda",
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path

        self.model, self.tokenizer = load_rewriter_model(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
            device=self.device,
        )

    def rewrite(
        self,
        query: str,
        context: Optional[str] = None,
        k: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Generate k rewritten queries for the given query and optional context."""
        start_time = time.time()
        prompt = format_prompt(query, context)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        prompt_len = input_ids.shape[1]

        # Handle greedy vs sampling
        if temperature <= 0.0 or k == 1:
            if temperature <= 0.0 and k > 1:
                logger.warning(
                    f"k={k} requested but temperature={0.0} forces greedy decode. Only 1 sequence will be returned."
                )
            do_sample = False
            num_seqs = 1
            temp_val = None
            top_p_val = None
            top_k_val = None
        else:
            do_sample = True
            num_seqs = k
            temp_val = max(temperature, 0.1)
            top_p_val = top_p
            top_k_val = top_k

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_seqs,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp_val
            gen_kwargs["top_p"] = top_p_val
            gen_kwargs["top_k"] = top_k_val

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        elapsed_ms = (time.time() - start_time) * 1000.0

        raw_candidates = []
        for out in outputs:
            generated_tokens = out[prompt_len:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            raw_candidates.append(text)

        cleaned_candidates = deduplicate_queries(raw_candidates)

        # Fallback to original query if nothing was generated
        if not cleaned_candidates:
            cleaned_candidates = [query]

        return {
            "original_query": query,
            "context": context,
            "rewritten_queries": cleaned_candidates,
            "k_requested": k,
            "k_generated": len(cleaned_candidates),
            "latency_ms": round(elapsed_ms, 2),
            "device": str(self.model.device),
        }

def get_default_rewriter(adapter_dir: Optional[str] = None) -> SLMQueryRewriter:
    model_cfg = load_yaml(resolve_path("configs/model.yaml"))["model"]
    candidate_paths = [
        adapter_dir,
        resolve_path("experiments/qlora_qwen/best_adapter"),
        resolve_path("experiments/qlora_qwen/final_adapter"),
        resolve_path("experiments/qlora_qwen/latest_adapter"),
    ]
    adapter_path = None
    for path in candidate_paths:
        if path and os.path.exists(path):
            adapter_path = path
            break
    return SLMQueryRewriter(
        base_model_name=model_cfg["name"],
        adapter_path=adapter_path,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )
