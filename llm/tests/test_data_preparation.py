import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from transformers import AutoTokenizer

from src.data_preparation import (
    build_training_text,
    load_raw_examples,
    prepare_dataset,
    split_examples,
    tokenize_example,
)

SEED_DATASET = Path(__file__).resolve().parent.parent / "data" / "seed_dataset.jsonl"


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_load_raw_examples_parses_seed_dataset():
    examples = load_raw_examples(SEED_DATASET)
    assert len(examples) > 0
    assert examples[0].input_text
    assert "date_of_occurrence" in examples[0].fir


def test_split_examples_is_deterministic_and_disjoint():
    examples = load_raw_examples(SEED_DATASET)
    split_a = split_examples(examples, val_ratio=0.2, seed=1)
    split_b = split_examples(examples, val_ratio=0.2, seed=1)
    assert [e.input_text for e in split_a["train"]] == [e.input_text for e in split_b["train"]]

    # Records themselves (input_text + fir payload) must not appear in
    # both splits, even though two distinct records can coincidentally
    # share the same short input_text (e.g. same phrase, different date).
    def record_key(e):
        return (e.input_text, json.dumps(e.fir, sort_keys=True, ensure_ascii=False))

    train_keys = {record_key(e) for e in split_a["train"]}
    val_keys = {record_key(e) for e in split_a["val"]}
    assert train_keys.isdisjoint(val_keys)
    assert len(split_a["train"]) + len(split_a["val"]) == len(examples)


def test_build_training_text_contains_prompt_and_json_completion(tokenizer):
    examples = load_raw_examples(SEED_DATASET)
    parts = build_training_text(tokenizer, examples[0])
    assert examples[0].input_text in parts["prompt"]
    completion_json = json.loads(parts["completion"].replace(tokenizer.eos_token or "", ""))
    assert completion_json["offense_type"] == examples[0].fir["offense_type"]


def test_tokenize_example_masks_prompt_tokens(tokenizer):
    examples = load_raw_examples(SEED_DATASET)
    # gpt2's BPE vocab has poor Bangla coverage (falls back to byte-level
    # tokens), so a generous max_length is used here to accommodate the
    # fallback prompt template without truncating the completion.
    tokenized = tokenize_example(tokenizer, examples[0], max_length=1024)
    assert len(tokenized["input_ids"]) == len(tokenized["labels"]) == len(tokenized["attention_mask"])
    # At least the leading prompt tokens must be masked with -100.
    assert tokenized["labels"][0] == -100
    # Some tokens (the completion) must not be masked.
    assert any(label != -100 for label in tokenized["labels"])


def test_prepare_dataset_writes_expected_files(tmp_path, tokenizer):
    out_dir = tmp_path / "processed"
    counts = prepare_dataset(SEED_DATASET, tokenizer, out_dir, val_ratio=0.2, max_length=256, seed=7)

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "val.jsonl").exists()
    assert (out_dir / "train_tokenized.jsonl").exists()
    assert (out_dir / "val_tokenized.jsonl").exists()
    assert counts["train"] > 0 and counts["val"] > 0

    with (out_dir / "train_tokenized.jsonl").open() as f:
        first_line = json.loads(f.readline())
    assert set(first_line.keys()) == {"input_ids", "attention_mask", "labels"}
