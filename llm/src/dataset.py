"""Dataset loading/tokenization for supervised fine-tuning (SFT) of the
BdSL-raw-text -> FIR-JSON model.

Each training example is tokenized as ``prompt_tokens + target_tokens`` and
loss is only computed on the ``target_tokens`` (the FIR JSON + EOS), i.e. the
model is never trained to "predict" its own instructions -- only to produce
the correct structured output for a given raw-text input.

Expects JSONL files where each line has at least:
    {"raw_signed_text": "...", "fir": {...}}
This is exactly what ``data/generate_synthetic_data.py`` produces, and what
``data/prepare_dataset.py`` should produce for real, collected FIR data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset

from .prompts import build_chat_messages, render_prompt_plain

IGNORE_INDEX = -100


def load_fir_jsonl(path: str) -> Dataset:
    return load_dataset("json", data_files=path, split="train")


def _prompt_ids(raw_signed_text: str, tokenizer) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        messages = build_chat_messages(raw_signed_text)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        prompt_text = render_prompt_plain(raw_signed_text)
        ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    # Depending on the transformers version, apply_chat_template may return a
    # plain list of ids or a BatchEncoding/dict with an "input_ids" key.
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def encode_example(example: dict[str, Any], tokenizer, max_length: int = 1024) -> dict[str, list[int]]:
    # `datasets` infers Arrow types from the JSONL values, so ISO date strings
    # (e.g. "date_of_occurrence") can come back as `datetime.date` objects
    # rather than `str`. `default=str` keeps serialization robust either way.
    target_json = json.dumps(example["fir"], ensure_ascii=False, default=str)

    prompt_ids = _prompt_ids(example["raw_signed_text"], tokenizer)
    target_ids = tokenizer(target_json, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)

    if len(input_ids) > max_length:
        # Trim from the left of the prompt first; keep the full target so the
        # model always sees a complete, well-formed JSON label to learn from.
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_sft_dataset(jsonl_path: str, tokenizer, max_length: int = 1024) -> Dataset:
    ds = load_fir_jsonl(jsonl_path)
    return ds.map(
        lambda ex: encode_example(ex, tokenizer, max_length),
        remove_columns=ds.column_names,
    )


@dataclass
class FIRDataCollator:
    """Dynamically pads ``input_ids``/``attention_mask``/``labels`` batches."""

    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch_input_ids, batch_attention, batch_labels = [], [], []
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_id] * n_pad)
            batch_attention.append(f["attention_mask"] + [0] * n_pad)
            batch_labels.append(f["labels"] + [IGNORE_INDEX] * n_pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
