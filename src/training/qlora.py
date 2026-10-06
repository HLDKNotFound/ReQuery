from __future__ import annotations
import os
import sys
import json
import time
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from src.utils.logging import get_logger
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_seed
from src.data.tokenizer import QueryRewriterCollator
from src.training.callbacks import MetricsAndVRAMCallback, BestAndLatestModelCallback
from src.training.checkpoint import save_adapter

logger = get_logger("training.qlora")

def train_qlora(
    model_config_path: str = "configs/model.yaml",
    training_config_path: str = "configs/training.yaml",
    train_data_path: str = "data/processed/train.jsonl",
    val_data_path: str = "data/processed/val.jsonl",
    output_dir: str | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    grad_accum_override: int | None = None,
    resume_from_checkpoint: str | bool | None = None,
):
    model_cfg = load_yaml(resolve_path(model_config_path))["model"]
    full_train_yaml = load_yaml(resolve_path(training_config_path))
    train_cfg = full_train_yaml["training"]
    lora_cfg = full_train_yaml["lora"]

    set_seed(train_cfg.get("seed", 42))

    effective_output_dir = output_dir or train_cfg.get("output_dir", "experiments/qlora_qwen")
    if effective_output_dir.endswith("/checkpoints"):
        resolved_output = resolve_path(os.path.dirname(effective_output_dir))
        checkpoints_dir = resolve_path(effective_output_dir)
    else:
        resolved_output = resolve_path(effective_output_dir)
        checkpoints_dir = os.path.join(resolved_output, "checkpoints")

    best_adapter_output = os.path.join(resolved_output, "best_adapter")
    latest_adapter_output = os.path.join(resolved_output, "latest_adapter")
    final_adapter_output = os.path.join(resolved_output, "final_adapter")
    os.makedirs(resolved_output, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    model_name = model_cfg["name"]
    logger.info(f"Loading base SLM for 4-bit QLoRA: {model_name}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("quantization_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=model_cfg.get("use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Prepare model for LoRA
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )

    # Attach PEFT LoRA adapter
    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    peft_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,} / {all_param:,} "
        f"({100 * trainable_params / all_param:.2f}%)"
    )

    # Load datasets
    train_file = resolve_path(train_data_path)
    val_file = resolve_path(val_data_path)

    raw_datasets = load_dataset("json", data_files={"train": train_file, "val": val_file})
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["val"]

    if max_train_samples and max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_val_samples and max_val_samples < len(val_dataset):
        val_dataset = val_dataset.select(range(max_val_samples))

    logger.info(f"Loaded {len(train_dataset):,} train examples and {len(val_dataset):,} val examples.")

    collator = QueryRewriterCollator(
        tokenizer,
        max_seq_length=model_cfg.get("max_seq_length", 512),
    )

    epochs = epochs_override if epochs_override is not None else train_cfg.get("epochs", 3)
    batch_size = batch_size_override if batch_size_override is not None else train_cfg.get("per_device_train_batch_size", 2)
    grad_accum = grad_accum_override if grad_accum_override is not None else train_cfg.get("gradient_accumulation_steps", 8)

    # -----------------------------------------------------------------
    # Checkpoint Discovery & Resuming
    # -----------------------------------------------------------------
    last_checkpoint = get_last_checkpoint(checkpoints_dir) if os.path.exists(checkpoints_dir) else None
    auto_resume = train_cfg.get("auto_resume", True)
    resume_param = resume_from_checkpoint
    if resume_param is None:
        resume_param = train_cfg.get("resume_from_checkpoint", auto_resume)

    checkpoint_to_resume = None
    if isinstance(resume_param, str) and resume_param.lower() not in ("auto", "true", "false", "none"):
        resolved_ckpt = resolve_path(resume_param)
        if os.path.exists(resolved_ckpt):
            checkpoint_to_resume = resolved_ckpt
            logger.info(f"Resuming training from explicit checkpoint: {checkpoint_to_resume}")
        else:
            logger.warning(f"Specified checkpoint '{resume_param}' not found. Starting fresh run.")
    elif resume_param is True or resume_param == "auto" or (isinstance(resume_param, str) and resume_param.lower() in ("true", "auto")):
        if last_checkpoint is not None:
            checkpoint_to_resume = last_checkpoint
            logger.info(
                f"Discovered previous latest checkpoint: {last_checkpoint}. "
                f"Resuming training (target epochs: {epochs})..."
            )
        else:
            logger.info(f"No existing checkpoint found in '{checkpoints_dir}'. Starting fresh training run.")
    else:
        logger.info("Resume disabled. Starting fresh training run.")

    # -----------------------------------------------------------------
    # Training Arguments: Best Model on Validation Set & Checkpoint Limits
    # -----------------------------------------------------------------
    eval_strategy = train_cfg.get("evaluation_strategy", "steps")
    save_strategy = train_cfg.get("save_strategy", eval_strategy)
    eval_steps = train_cfg.get("eval_steps", 50)
    save_steps = train_cfg.get("save_steps", 50)
    load_best = train_cfg.get("load_best_model_at_end", True)
    metric_best = train_cfg.get("metric_for_best_model", "eval_loss")
    greater_best = train_cfg.get("greater_is_better", False)

    has_eval_dataset = val_dataset is not None and len(val_dataset) > 0
    actual_load_best = load_best and has_eval_dataset
    actual_eval_strategy = eval_strategy if has_eval_dataset else "no"

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=actual_eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=actual_load_best,
        metric_for_best_model=metric_best if actual_load_best else None,
        greater_is_better=greater_best if actual_load_best else None,
        bf16=train_cfg.get("bf16", True) and torch.cuda.is_available(),
        fp16=train_cfg.get("fp16", False),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        report_to="none",
        remove_unused_columns=False,
    )

    metrics_cb = MetricsAndVRAMCallback(log_file=os.path.join(resolved_output, "train.log"))
    latest_cb = BestAndLatestModelCallback(tokenizer=tokenizer, latest_adapter_dir=latest_adapter_output)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if has_eval_dataset else None,
        data_collator=collator,
        callbacks=[metrics_cb, latest_cb],
    )

    logger.info("Starting QLoRA fine-tuning...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    training_time = time.time() - start_time

    # -----------------------------------------------------------------
    # Save Best and Latest Model Adapters
    # -----------------------------------------------------------------
    # 1. Best model: When load_best_model_at_end=True, trainer.model has the weights from best validation checkpoint
    save_adapter(model, tokenizer, best_adapter_output)
    logger.info(f"Saved best validation model adapter to: {best_adapter_output}")

    # 2. Final adapter: Maintain final_adapter for backward compatibility
    save_adapter(model, tokenizer, final_adapter_output)
    logger.info(f"Saved final adapter (best model) to: {final_adapter_output}")

    # 3. Ensure latest adapter exists on disk
    if not os.path.exists(latest_adapter_output):
        save_adapter(model, tokenizer, latest_adapter_output)
        logger.info(f"Saved latest model adapter to: {latest_adapter_output}")

    # Save summary metrics
    metrics = {
        "train_runtime_sec": round(training_time, 2),
        "train_samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 3),
        "train_loss": round(train_result.metrics.get("train_loss", 0), 4),
        "trainable_parameters": trainable_params,
        "total_parameters": all_param,
        "trainable_percent": round(100 * trainable_params / all_param, 4),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0,
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "best_metric": getattr(trainer.state, "best_metric", None),
        "metric_name": metric_best if actual_load_best else None,
        "latest_checkpoint": get_last_checkpoint(checkpoints_dir),
        "completed_epochs": round(trainer.state.epoch or 0.0, 3),
        "global_step": trainer.state.global_step,
    }

    metrics_path = os.path.join(resolved_output, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training completed in {training_time:.1f}s. Final metrics: {metrics}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SLM for RAG Query Rewriting with QLoRA")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (increase to continue training)")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit number of train samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit number of validation samples")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default="auto",
        help="Resume from previous checkpoint ('auto' or specific checkpoint path)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh training run, ignoring existing checkpoints",
    )
    args = parser.parse_args()

    resume_param = False if args.no_resume else (args.resume if args.resume is not None else "auto")

    train_qlora(
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        grad_accum_override=args.grad_accum,
        resume_from_checkpoint=resume_param,
    )

if __name__ == "__main__":
    main()
