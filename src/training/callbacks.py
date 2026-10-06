from __future__ import annotations
import os
import time
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from src.utils.logging import get_logger

logger = get_logger("training.callbacks")

class MetricsAndVRAMCallback(TrainerCallback):
    """Callback to log loss, learning rate, and GPU VRAM usage at each logging step."""

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        self.start_time = time.time()
        if self.log_file:
            parent = os.path.dirname(self.log_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("step,epoch,loss,eval_loss,lr,vram_allocated_mb,vram_reserved_mb,elapsed_sec\n")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        epoch = round(state.epoch or 0.0, 3)
        loss = logs.get("loss", "")
        eval_loss = logs.get("eval_loss", "")
        lr = logs.get("learning_rate", "")

        vram_alloc = round(torch.cuda.memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0
        vram_res = round(torch.cuda.memory_reserved() / (1024**2), 1) if torch.cuda.is_available() else 0.0
        elapsed = round(time.time() - self.start_time, 1)

        msg = f"Step {step} | Epoch {epoch} | Loss: {loss} | VRAM: {vram_alloc}MB / {vram_res}MB | Time: {elapsed}s"
        logger.info(msg)

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{step},{epoch},{loss},{eval_loss},{lr},{vram_alloc},{vram_res},{elapsed}\n")

class BestAndLatestModelCallback(TrainerCallback):
    """Callback to continuously update the latest adapter checkpoint and track training progress."""

    def __init__(self, tokenizer, latest_adapter_dir: str):
        self.tokenizer = tokenizer
        self.latest_adapter_dir = latest_adapter_dir

    def _save_latest(self, model, reason: str = ""):
        if model is not None:
            from src.training.checkpoint import save_adapter
            parent = os.path.dirname(self.latest_adapter_dir)
            if parent:
                os.makedirs(parent, exist_ok=True)
            save_adapter(model, self.tokenizer, self.latest_adapter_dir)
            if reason:
                logger.info(f"Updated latest model adapter at {self.latest_adapter_dir} ({reason})")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        self._save_latest(model, f"checkpoint saved at step {state.global_step}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        self._save_latest(model, f"epoch {round(state.epoch or 0.0, 2)} finished")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Save on the final step before load_best_model_at_end can swap weights
        if control.should_training_stop or (state.max_steps > 0 and state.global_step >= state.max_steps):
            model = kwargs.get("model")
            self._save_latest(model, f"final step {state.global_step} reached")

