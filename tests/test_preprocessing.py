from __future__ import annotations
import pytest
from src.data.preprocess import clean_text, parse_canard_record, parse_qrecc_record, parse_ambigqa_record

def test_clean_text():
    raw = "  What   did    the paper \n say?   "
    cleaned = clean_text(raw)
    assert cleaned == "What did the paper say?"

def test_parse_canard():
    raw_record = {
        "cur_question": "What other languages are spoken there?",
        "prev_questions": "Where is Malayali located?",
        "answer_text_with_window": "What other languages are spoken in Malayali?",
        "answer_text": "Malayalam is widely spoken.",
        "id": "sample_101",
    }
    parsed = parse_canard_record(raw_record, "train")
    assert parsed is not None
    assert parsed["query"] == "What other languages are spoken there?"
    assert parsed["rewrite"] == "What other languages are spoken in Malayali?"
    assert parsed["context"] == "Where is Malayali located?"
    assert parsed["dataset"] == "canard"

def test_parse_empty_filter():
    raw_record = {"cur_question": "", "answer_text_with_window": ""}
    assert parse_canard_record(raw_record, "train") is None
