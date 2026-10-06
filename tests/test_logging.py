from __future__ import annotations
import os
import tempfile
from src.utils.logging import get_logger

def test_logger_creation_and_file_handler():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_log_dir", "sub", "app.log")
        logger = get_logger("test_module", log_file=log_file)

        test_msg = "ReQuery test log message 12345"
        logger.info(test_msg)

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert test_msg in content
