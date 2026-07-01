"""Build tokenized, instruction-tuning-ready datasets for the Bangla-FIR
LLM from raw (input_text, fir) JSONL records.

Example raw record (see ``data/seed_dataset.jsonl``):

    {"input_text": "আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকাল ৫টায়",
     "fir": {"date_of_occurrence": "...", ...}}

This module turns each record into a full prompt+completion string
using the shared prompt template (``src/prompts.py``) and the FIR JSON
schema (``src/fir_schema.py``), tokenizes it, and masks the prompt
tokens out of the loss so the model is only trained to predict the
completion (the FIR JSON).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from .fir_schema import FIRRecord
    from .prompts import build_prompt
except ImportError:  # pragma: no cover - allows running as a script
    from fir_schema import FIRRecord
    from prompts import build_prompt


@dataclass
class RawExample:
    input_text: str
    fir: Dict[str, str]


def load_raw_examples(path: Path) -> List[RawExample]:
    examples: List[RawExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if "input_text" not in obj or "fir" not in obj:
                raise ValueError(f"{path}:{line_no}: record missing 'input_text' or 'fir' key")
            examples.append(RawExample(input_text=obj["input_text"], fir=obj["fir"]))
    return examples


def split_examples(
    examples: List[RawExample], val_ratio: float = 0.15, seed: int = 42
) -> Dict[str, List[RawExample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    return {"train": shuffled[n_val:], "val": shuffled[:n_val]}


def build_training_text(tokenizer, example: RawExample) -> Dict[str, str]:
    """Return the prompt and target-completion strings for one example."""

    record = FIRRecord.from_dict(example.fir)
    prompt = build_prompt(tokenizer, example.input_text)
    completion = record.to_json(indent=2)
    eos = tokenizer.eos_token or ""
    return {"prompt": prompt, "completion": completion + eos}


def tokenize_example(tokenizer, example: RawExample, max_length: int = 768) -> Dict[str, List[int]]:
    """Tokenize a single example with prompt tokens masked out of the loss.

    Returns a dict with ``input_ids``, ``attention_mask`` and ``labels``
    (labels use -100 for prompt tokens, per HF convention for ignored
    positions in the cross-entropy loss).
    """

    parts = build_training_text(tokenizer, example)
    prompt_ids = tokenizer(parts["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(parts["completion"], add_special_tokens=False)["input_ids"]

    # The completion (target FIR JSON) is what the model is trained to
    # predict, so it must never be silently dropped by truncation. If
    # the combined sequence exceeds max_length, truncate the (masked)
    # prompt from the left first, and only trim the completion as a
    # last resort for pathologically long targets.
    completion_ids = completion_ids[:max_length]
    available_for_prompt = max(0, max_length - len(completion_ids))
    prompt_ids = prompt_ids[-available_for_prompt:] if available_for_prompt else []

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def prepare_dataset(
    raw_path: Path,
    tokenizer,
    output_dir: Path,
    val_ratio: float = 0.15,
    max_length: int = 768,
    seed: int = 42,
) -> Dict[str, int]:
    """End-to-end: load raw JSONL, split, tokenize, and persist to disk.

    Writes ``train.jsonl`` and ``val.jsonl`` (untokenized, human
    readable, for inspection/eval) plus ``train_tokenized.jsonl`` and
    ``val_tokenized.jsonl`` (tokenized, ready for the ``Trainer``) into
    ``output_dir``. Returns example counts per split.
    """

    examples = load_raw_examples(raw_path)
    splits = split_examples(examples, val_ratio=val_ratio, seed=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for split_name, split_examples_ in splits.items():
        readable_path = output_dir / f"{split_name}.jsonl"
        tokenized_path = output_dir / f"{split_name}_tokenized.jsonl"
        with readable_path.open("w", encoding="utf-8") as readable_f, tokenized_path.open(
            "w", encoding="utf-8"
        ) as tok_f:
            for ex in split_examples_:
                readable_f.write(
                    json.dumps({"input_text": ex.input_text, "fir": ex.fir}, ensure_ascii=False) + "\n"
                )
                tokenized = tokenize_example(tokenizer, ex, max_length=max_length)
                tok_f.write(json.dumps(tokenized) + "\n")
        counts[split_name] = len(split_examples_)

    return counts


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "seed_dataset.jsonl")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "processed")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name/path used to pre-tokenize.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    counts = prepare_dataset(
        args.raw, tokenizer, args.out, val_ratio=args.val_ratio, max_length=args.max_length, seed=args.seed
    )
    print(f"Prepared dataset from {args.raw} -> {args.out}: {counts}")


if __name__ == "__main__":
    _cli()
