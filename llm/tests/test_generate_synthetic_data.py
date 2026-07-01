import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from generate_synthetic_data import generate_dataset  # noqa: E402
from src.prompts import REQUIRED_FIR_KEYS  # noqa: E402


def test_generate_dataset_size_and_ids():
    records = generate_dataset(n=25, seed=7)
    assert len(records) == 25
    ids = [r["id"] for r in records]
    assert len(set(ids)) == 25


def test_generate_dataset_is_deterministic_given_seed():
    a = generate_dataset(n=10, seed=123)
    b = generate_dataset(n=10, seed=123)
    assert a == b


def test_generate_dataset_records_have_required_fir_keys():
    records = generate_dataset(n=50, seed=3)
    for rec in records:
        for key in REQUIRED_FIR_KEYS:
            assert key in rec["fir"], f"missing {key} in record {rec['id']}"
        assert rec["raw_signed_text"].strip()
        assert rec["fir"]["narrative_bn"].strip()


def test_generate_dataset_no_unresolved_placeholders():
    records = generate_dataset(n=60, seed=99)
    for rec in records:
        assert "{" not in rec["raw_signed_text"]
        assert "{" not in rec["fir"]["narrative_bn"]


def test_missing_person_records_use_not_applicable_accused():
    records = generate_dataset(n=200, seed=5)
    missing = [r for r in records if r["offense_type_key"] == "missing_person"]
    assert missing, "expected at least one missing_person record in a sample of 200"
    for rec in missing:
        assert rec["fir"]["accused_name"] == "প্রযোজ্য নয়"
        assert rec["fir"]["items_involved"] == []
