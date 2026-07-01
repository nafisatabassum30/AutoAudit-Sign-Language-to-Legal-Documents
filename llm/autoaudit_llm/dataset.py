# -*- coding: utf-8 -*-
"""Dataset + collation utilities for instruction fine-tuning of the
FIR-generation LLM.

Only the assistant's completion (the target JSON) contributes to the loss;
the system + user turns are masked out. This is implemented manually (rather
than relying on a specific chat-template convention) so the same code works
whether the base model ships with a proper chat template (Qwen, Llama-3,
...) or not (useful for cheap smoke tests with tiny models).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


def format_messages(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """Render a list of chat messages to text, using the tokenizer's chat
    template when available and falling back to a simple ChatML-like format
    otherwise (e.g. for base models without an instruction chat template).
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    parts = []
    for message in messages:
        parts.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


class FIRInstructionDataset(Dataset):
    """Reads the JSONL produced by ``data/prepare_dataset.py`` (rows with a
    ``messages`` field: system/user/assistant turns) and tokenizes them with
    the assistant completion as the only supervised span.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.rows: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        messages = self.rows[idx]["messages"]
        prompt_messages = messages[:-1]

        full_text = format_messages(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = format_messages(self.tokenizer, prompt_messages, add_generation_prompt=True)

        eos_id = self.tokenizer.eos_token_id
        full_ids: List[int] = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_length, add_special_tokens=False
        )["input_ids"]
        if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
            full_ids = full_ids[: self.max_seq_length - 1] + [eos_id]

        prompt_ids: List[int] = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_length, add_special_tokens=False
        )["input_ids"]

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


class PadCollator:
    """Right-pads a batch of variable-length tokenized examples."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(seq: List[int], pad_value: int) -> List[int]:
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features])
        attention_mask = torch.tensor([pad(f["attention_mask"], 0) for f in features])
        labels = torch.tensor([pad(f["labels"], self.label_pad_token_id) for f in features])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
