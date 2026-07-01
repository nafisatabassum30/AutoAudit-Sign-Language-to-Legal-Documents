"""Tests for FIRInstructionDataset / PadCollator that avoid any network calls
by using a tiny deterministic fake tokenizer instead of downloading a real
HF model.
"""
import json
import os
import tempfile

from autoaudit_llm.dataset import FIRInstructionDataset, PadCollator, format_messages


class DummyTokenizer:
    """Whitespace tokenizer with a stable, made-up vocabulary. No network,
    no chat_template -- exercises the plain ChatML-like fallback formatting
    path in ``format_messages``.
    """

    chat_template = None
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=10_000, add_special_tokens=False):
        tokens = text.split()
        ids = [abs(hash(tok)) % 50000 + 2 for tok in tokens]  # +2 to dodge special ids 0/1
        ids = ids[:max_length]
        return {"input_ids": ids}


def _write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_row():
    messages = [
        {"role": "system", "content": "তুমি একজন সহকারী।"},
        {"role": "user", "content": "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫"},
        {"role": "assistant", "content": '{"offense_type": "চুরি"}'},
    ]
    return {"raw_text": "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫", "target_json": messages[-1]["content"], "messages": messages}


def test_format_messages_fallback_includes_all_turns():
    tokenizer = DummyTokenizer()
    messages = _sample_row()["messages"]
    text = format_messages(tokenizer, messages, add_generation_prompt=False)
    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "চুরি" in text


def test_dataset_masks_prompt_and_supervises_completion_only():
    tokenizer = DummyTokenizer()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "train.jsonl")
        _write_jsonl([_sample_row()], path)
        dataset = FIRInstructionDataset(path, tokenizer, max_seq_length=1000)
        item = dataset[0]

        labels = item["labels"]
        input_ids = item["input_ids"]
        assert len(labels) == len(input_ids)

        # There must be a masked prefix (system+user) ...
        assert labels[0] == -100
        # ... followed by at least one supervised (non -100) token for the
        # assistant completion.
        assert any(label != -100 for label in labels)
        # And the supervised region should be a suffix, not scattered.
        first_supervised = next(i for i, l in enumerate(labels) if l != -100)
        assert all(l == -100 for l in labels[:first_supervised])
        assert all(l != -100 for l in labels[first_supervised:])


def test_pad_collator_produces_equal_length_batches():
    tokenizer = DummyTokenizer()
    features = [
        {"input_ids": [2, 3, 4], "attention_mask": [1, 1, 1], "labels": [-100, -100, 4]},
        {"input_ids": [2, 3], "attention_mask": [1, 1], "labels": [-100, 3]},
    ]
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"][1][-1].item() == -100  # padded label slot
    assert batch["attention_mask"][1][-1].item() == 0
