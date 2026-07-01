"""Dataset I/O and formatting for instruction fine-tuning.

Responsibilities:

* Read/write JSONL example files (``{"raw_statement", "extraction",
  "complaint"}`` records).
* Convert those records into the prompt/completion text used by the trainer.
* Split a dataset into train/eval partitions.

The formatting relies on :mod:`autoaudit_llm.prompts` so the exact same layout
is used at train and inference time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .prompts import build_chat_messages, build_training_text, target_json
from .schema import FIRComplaint


def write_jsonl(records: List[Dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_dataset(
    records: List[Dict], eval_split: float = 0.1, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle and split records into (train, eval)."""
    if not 0.0 <= eval_split < 1.0:
        raise ValueError("eval_split must be in [0, 1)")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_split)
    return shuffled[n_eval:], shuffled[:n_eval]


def record_to_complaint(record: Dict) -> FIRComplaint:
    return FIRComplaint(**record["complaint"])


def format_for_training(
    record: Dict, tokenizer: Optional[object] = None
) -> str:
    """Format a single record into a training text string.

    If a HuggingFace tokenizer with a chat template is supplied, the chat
    template is used (recommended for instruct models). Otherwise a simple
    ``<|system|>/<|user|>/<|assistant|>`` layout is produced.
    """
    complaint = record_to_complaint(record)
    raw = record["raw_statement"]

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = build_chat_messages(raw)
        messages.append({"role": "assistant", "content": target_json(complaint)})
        return tokenizer.apply_chat_template(messages, tokenize=False)

    return build_training_text(raw, complaint)


def build_hf_dataset(records: List[Dict], tokenizer: Optional[object] = None):
    """Return a :class:`datasets.Dataset` with a single ``text`` column.

    Imported lazily so the package works without the (heavy) ``datasets`` dep
    for non-training use cases.
    """
    from datasets import Dataset  # type: ignore

    texts = [format_for_training(rec, tokenizer) for rec in records]
    return Dataset.from_dict({"text": texts})
