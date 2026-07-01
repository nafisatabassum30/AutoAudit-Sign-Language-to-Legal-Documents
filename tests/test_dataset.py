from autoaudit_llm.data_generation import generate_dataset
from autoaudit_llm.dataset import (
    format_for_training,
    read_jsonl,
    record_to_complaint,
    split_dataset,
    write_jsonl,
)


def test_write_and_read_roundtrip(tmp_path):
    records = generate_dataset(10, seed=2)
    path = tmp_path / "data.jsonl"
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert loaded == records


def test_split_dataset_sizes():
    records = generate_dataset(100, seed=5)
    train, eval_ = split_dataset(records, eval_split=0.2, seed=5)
    assert len(eval_) == 20
    assert len(train) == 80
    # disjoint
    train_keys = {r["raw_statement"] for r in train}
    eval_keys = {r["raw_statement"] for r in eval_}
    assert train_keys.isdisjoint(eval_keys)


def test_format_for_training_plain_layout():
    record = generate_dataset(1, seed=9)[0]
    text = format_for_training(record, tokenizer=None)
    assert "<|system|>" in text
    assert "<|assistant|>" in text
    # completion contains the offense type
    assert record["complaint"]["offense_type"] in text


def test_record_to_complaint():
    record = generate_dataset(1, seed=11)[0]
    complaint = record_to_complaint(record)
    assert complaint.offense_type == record["complaint"]["offense_type"]
