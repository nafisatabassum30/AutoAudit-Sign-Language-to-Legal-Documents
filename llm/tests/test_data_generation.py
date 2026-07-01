import json
from pathlib import Path

from src.data_generation import build_dataset, generate_examples
from src.schema import FIRRecord


def test_generate_examples_are_schema_valid():
    examples = generate_examples(50, seed=1)
    assert len(examples) == 50
    for ex in examples:
        assert "input_text" in ex and ex["input_text"]
        record = FIRRecord.model_validate(ex["output_json"])
        assert record.incident_location
        assert record.narrative_bn


def test_generate_examples_deterministic_with_seed():
    a = generate_examples(20, seed=7)
    b = generate_examples(20, seed=7)
    assert a == b


def test_generate_examples_diverse_offense_types():
    examples = generate_examples(200, seed=3)
    offense_types = {ex["output_json"]["offense_type"] for ex in examples}
    assert len(offense_types) >= 4


def test_build_dataset_writes_splits(tmp_path: Path):
    counts = build_dataset(100, tmp_path, seed=5, val_frac=0.1, test_frac=0.1)
    assert counts["train"] + counts["val"] + counts["test"] == 100
    for split in ("train", "val", "test"):
        out_file = tmp_path / f"{split}.jsonl"
        assert out_file.exists()
        lines = out_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == counts[split]
        for line in lines:
            row = json.loads(line)
            FIRRecord.model_validate(row["output_json"])


def test_build_dataset_mixes_in_extra_jsonl(tmp_path: Path):
    extra_path = tmp_path / "real_examples.jsonl"
    example = generate_examples(1, seed=99)[0]
    extra_path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    counts = build_dataset(10, out_dir, seed=5, extra_jsonl_paths=[extra_path])
    assert sum(counts.values()) == 11
