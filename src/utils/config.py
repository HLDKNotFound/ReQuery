from __future__ import annotations
import os
import yaml
from typing import Any, Dict

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resolve_path(relative_or_absolute_path: str) -> str:
    """Resolve a path relative to the project root if it is not absolute."""
    if os.path.isabs(relative_or_absolute_path):
        return relative_or_absolute_path
    return os.path.join(get_project_root(), relative_or_absolute_path)
