"""Load the JSONL train/val/test splits and format them for SFT training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .prompts import build_training_example


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_chat_dataset(rows: List[Dict]) -> List[Dict]:
    """Convert raw ``{input_text, output_json}`` rows into ``{messages: [...]}``
    records ready for ``trl.SFTTrainer`` (which accepts a ``messages`` column
    and applies the tokenizer's chat template automatically).
    """
    formatted = []
    for row in rows:
        messages = build_training_example(row["input_text"], row["output_json"])
        formatted.append({"messages": messages})
    return formatted


def load_hf_dataset(data_dir: Path, split: str):
    """Load a split as a :class:`datasets.Dataset` (imports ``datasets`` lazily
    so the rest of this module works without the heavy training deps
    installed, e.g. for tests)."""
    from datasets import Dataset

    rows = load_jsonl(Path(data_dir) / f"{split}.jsonl")
    chat_rows = to_chat_dataset(rows)
    return Dataset.from_list(chat_rows)
