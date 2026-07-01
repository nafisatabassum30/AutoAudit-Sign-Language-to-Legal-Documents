"""Utilities for preparing supervised fine-tuning datasets."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from prompts import SYSTEM_PROMPT, build_user_prompt


def _to_chat_example(record: dict[str, Any]) -> dict[str, str]:
    sign_text_bn = record["sign_text_bn"]
    target = record["target"]
    metadata = record.get("metadata", "অজানা")

    assistant_text = json.dumps(target, ensure_ascii=False)
    user_text = build_user_prompt(sign_text_bn=sign_text_bn, metadata=json.dumps(metadata, ensure_ascii=False))

    text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_text}\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{assistant_text}\n<|eot_id|>"
    )

    return {"text": text}


def load_sft_dataset(jsonl_path: str) -> Dataset:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "sign_text_bn" not in obj or "target" not in obj:
                raise ValueError(f"Missing required keys at line {line_num}")
            rows.append(_to_chat_example(obj))

    if not rows:
        raise ValueError("No valid rows found in dataset.")
    return Dataset.from_list(rows)
