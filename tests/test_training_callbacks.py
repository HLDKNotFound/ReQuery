from __future__ import annotations
import os
import tempfile
from unittest.mock import MagicMock
from transformers import TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint
from src.training.callbacks import BestAndLatestModelCallback

def test_best_and_latest_model_callback():
    with tempfile.TemporaryDirectory() as tmpdir:
        latest_adapter_dir = os.path.join(tmpdir, "latest_adapter")
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        cb = BestAndLatestModelCallback(tokenizer=mock_tokenizer, latest_adapter_dir=latest_adapter_dir)

        args = MagicMock(spec=TrainingArguments)
        state = MagicMock(spec=TrainerState)
        state.global_step = 10
        state.epoch = 1.0
        state.max_steps = 20
        control = MagicMock(spec=TrainerControl)
        control.should_training_stop = False

        # Test on_save
        cb.on_save(args, state, control, model=mock_model)
        assert mock_model.save_pretrained.called
        assert mock_tokenizer.save_pretrained.called

        mock_model.reset_mock()
        mock_tokenizer.reset_mock()

        # Test on_epoch_end
        cb.on_epoch_end(args, state, control, model=mock_model)
        assert mock_model.save_pretrained.called

        mock_model.reset_mock()

        # Test on_step_end when step limit is reached
        state.global_step = 20
        cb.on_step_end(args, state, control, model=mock_model)
        assert mock_model.save_pretrained.called

def test_checkpoint_discovery_for_resuming():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoints_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(os.path.join(checkpoints_dir, "checkpoint-50"))
        os.makedirs(os.path.join(checkpoints_dir, "checkpoint-100"))

        last = get_last_checkpoint(checkpoints_dir)
        assert last is not None
        assert last.endswith("checkpoint-100")
