from __future__ import annotations
import os
import pytest
from src.utils.config import load_yaml, resolve_path, get_project_root

def test_resolve_path_and_project_root():
    root = get_project_root()
    assert os.path.exists(root)
    assert os.path.isdir(root)

    abs_p = resolve_path("configs/model.yaml")
    assert os.path.isabs(abs_p)
    assert os.path.exists(abs_p)

    already_abs = os.path.abspath(__file__)
    assert resolve_path(already_abs) == already_abs

def test_load_yaml_valid():
    cfg = load_yaml(resolve_path("configs/model.yaml"))
    assert "model" in cfg
    assert "name" in cfg["model"]

def test_load_yaml_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_yaml("nonexistent_path_to_config.yaml")
