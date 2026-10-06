from __future__ import annotations
import os
import json
import shutil
from typing import Dict, Any
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from src.utils.logging import get_logger
from src.utils.config import resolve_path

logger = get_logger("data.download")

def download_qrecc(raw_dir: str) -> Dict[str, str]:
    """Download QReCC conversational query rewriting dataset from Hugging Face."""
    logger.info("Downloading QReCC dataset...")
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for filename in ["qrecc-training.json", "qrecc-test.json"]:
        dest = os.path.join(raw_dir, filename)
        if not os.path.exists(dest):
            downloaded = hf_hub_download(
                repo_id="svakulenk0/qrecc",
                filename=filename,
                repo_type="dataset",
            )
            shutil.copy2(downloaded, dest)
            logger.info(f"Downloaded QReCC {filename} to {dest}")
        else:
            logger.info(f"QReCC {filename} already exists at {dest}")
        paths[filename] = dest
    return paths

def download_canard(raw_dir: str) -> Dict[str, str]:
    """Download CANARD query rewriting dataset from Hugging Face."""
    logger.info("Downloading CANARD dataset...")
    os.makedirs(raw_dir, exist_ok=True)
    paths = {}
    for filename in ["train_gold_supervision.json", "dev_gold_supervision.json", "test_gold_supervision.json"]:
        dest = os.path.join(raw_dir, filename)
        if not os.path.exists(dest):
            downloaded = hf_hub_download(
                repo_id="uva-irlab/canard_quretec",
                filename=filename,
                repo_type="dataset",
            )
            shutil.copy2(downloaded, dest)
            logger.info(f"Downloaded CANARD {filename} to {dest}")
        else:
            logger.info(f"CANARD {filename} already exists at {dest}")
        paths[filename] = dest
    return paths

def download_ambigqa(raw_dir: str) -> str:
    """Download AmbigQA dataset for ambiguous query resolution."""
    logger.info("Downloading AmbigQA dataset...")
    dest = os.path.join(raw_dir, "ambigqa_train.json")
    if not os.path.exists(dest):
        try:
            ds = load_dataset("sewon/ambig_qa", split="train")
            records = []
            for item in ds:
                records.append(dict(item))
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(records, f)
            logger.info(f"Downloaded and saved {len(records)} AmbigQA examples to {dest}")
        except Exception as e:
            logger.warning(f"Could not download sewon/ambig_qa via datasets: {e}")
    else:
        logger.info(f"AmbigQA already exists at {dest}")
    return dest

def download_all(output_dir: str | None = None) -> Dict[str, Any]:
    raw_dir = output_dir or resolve_path("data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    qrecc_paths = download_qrecc(raw_dir)
    canard_paths = download_canard(raw_dir)
    ambigqa_path = download_ambigqa(raw_dir)
    return {
        "qrecc": qrecc_paths,
        "canard": canard_paths,
        "ambigqa": ambigqa_path,
    }

if __name__ == "__main__":
    download_all()
