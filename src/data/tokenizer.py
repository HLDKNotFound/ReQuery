from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.utils.logging import get_logger

logger = get_logger("data.tokenizer")

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert search engine query optimizer. "
    "Given a user query and optional conversation context, rewrite the query to make it "
    "specific, self-contained, disambiguated, and optimal for information retrieval."
)

def format_prompt(query: str, context: Optional[str] = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Format prompt using standard ChatML template."""
    user_content = ""
    if context and context.strip():
        user_content += f"Conversation context:\n{context.strip()}\n\n"
    user_content += f"Original query:\n{query.strip()}\n\nRewrite this query for optimal search retrieval:"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def format_target(rewrite: str) -> str:
    """Format assistant response target with trailing EOS."""
    return f"{rewrite.strip()}<|im_end|>"

class QueryRewriterCollator:
    """Data collator that handles dynamic padding and label masking (-100 for prompt)."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            query = item["query"]
            context = item.get("context")
            rewrite = item["rewrite"]

            prompt_str = format_prompt(query, context)
            target_str = format_target(rewrite)

            # Tokenize prompt and target separately to precisely locate boundary
            prompt_tokens = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            target_tokens = self.tokenizer.encode(target_str, add_special_tokens=False)

            combined_tokens = prompt_tokens + target_tokens
            if len(combined_tokens) > self.max_seq_length:
                # Truncate prompt from the left if needed, keep target intact if possible
                excess = len(combined_tokens) - self.max_seq_length
                prompt_tokens = prompt_tokens[excess:]
                combined_tokens = prompt_tokens + target_tokens

            # Labels: -100 for prompt tokens, target token ids for target
            labels = [-100] * len(prompt_tokens) + target_tokens

            input_ids_list.append(torch.tensor(combined_tokens, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_list.append(torch.ones(len(combined_tokens), dtype=torch.long))

        # Dynamic padding to longest sequence in batch
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for ids, lbl, att in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)]))
            padded_attention_masks.append(torch.cat([att, torch.zeros(pad_len, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_masks),
        }

def debug_tokenization(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Verify and print tokenization alignment for debugging."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    sample_query = "What did the paper say about training?"
    sample_context = "Discussion about the Transformer architecture in 'Attention Is All You Need'."
    sample_rewrite = "What training procedure and optimizer did Attention Is All You Need use?"

    prompt_str = format_prompt(sample_query, sample_context)
    target_str = format_target(sample_rewrite)

    collator = QueryRewriterCollator(tokenizer, max_seq_length=512)
    batch = [{"query": sample_query, "context": sample_context, "rewrite": sample_rewrite}]
    output = collator(batch)

    input_ids = output["input_ids"][0]
    labels = output["labels"][0]
    attention_mask = output["attention_mask"][0]

    # Decode masked vs unmasked
    unmasked_tokens = [tok for tok, lbl in zip(input_ids, labels) if lbl != -100]
    decoded_target = tokenizer.decode(unmasked_tokens)
    full_decoded = tokenizer.decode(input_ids)

    print("=" * 60)
    print("TOKENIZATION VERIFICATION REPORT")
    print("=" * 60)
    print("Original Query:", sample_query)
    print("Context:", sample_context)
    print("-" * 60)
    print("Prompt String:\n", prompt_str)
    print("Target String:\n", target_str)
    print("-" * 60)
    print("Input IDs (shape):", input_ids.shape)
    print("Labels (shape):", labels.shape)
    print("Masked Tokens Count (-100):", (labels == -100).sum().item())
    print("Target Tokens Count (Supervised):", (labels != -100).sum().item())
    print("-" * 60)
    print("Decoded Target (Only Loss-bearing tokens):")
    print(decoded_target)
    print("=" * 60)

if __name__ == "__main__":
    debug_tokenization()
