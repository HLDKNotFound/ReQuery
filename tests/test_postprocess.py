from __future__ import annotations
from src.inference.postprocess import clean_query_text, deduplicate_queries

def test_clean_query_text():
    # ChatML tags stripped
    raw = "<|im_start|>assistant\nWhat is quantum computing?<|im_end|>"
    assert clean_query_text(raw) == "What is quantum computing?"

    # Prefixes like 'Query: ' stripped
    raw2 = 'Query: "machine learning optimization"'
    assert clean_query_text(raw2) == "machine learning optimization"

    # Number bullets stripped
    raw3 = "1. how to train deep learning models"
    assert clean_query_text(raw3) == "how to train deep learning models"

    # Empty text
    assert clean_query_text("") == ""

def test_deduplicate_queries():
    candidates = [
        "What is quantum computing?",
        "what is quantum computing?",  # duplicate case-insensitively
        "Explain quantum computing basics",
        "",  # empty should be skipped
        "  ",
    ]
    deduped = deduplicate_queries(candidates)
    assert len(deduped) == 2
    assert deduped[0] == "What is quantum computing?"
    assert deduped[1] == "Explain quantum computing basics"
