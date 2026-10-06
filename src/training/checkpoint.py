from __future__ import annotations
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils.logging import get_logger

logger = get_logger("training.checkpoint")

def save_adapter(model, tokenizer, output_dir: str):
    """Save trained LoRA adapter and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Adapter and tokenizer saved successfully.")

def load_rewriter_model(
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
    device: str = "cuda",
):
    """Load base model (optionally quantized) and attach LoRA adapter if provided."""
    logger.info(f"Loading base model: {base_model_name} (load_in_4bit={load_in_4bit})")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    else:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=compute_dtype,
            device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if device == "cpu" or not torch.cuda.is_available():
            model = model.to("cpu")

    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("LoRA adapter attached successfully.")
    else:
        logger.info("No adapter path provided; using base model directly.")

    model.eval()
    return model, tokenizer
