import pytest
from pydantic import ValidationError

from src.schema import FIRRecord, OffenseType, Person, PropertyItem


def _minimal_record(**overrides):
    defaults = dict(
        offense_type=OffenseType.THEFT,
        complainant=Person(name="করিম উদ্দিন"),
        incident_location="উত্তরা",
        narrative_bn="একটি পরীক্ষামূলক বিবরণ।",
    )
    defaults.update(overrides)
    return FIRRecord(**defaults)


def test_minimal_record_valid():
    record = _minimal_record()
    assert record.offense_type == OffenseType.THEFT
    assert record.complainant.name == "করিম উদ্দিন"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        FIRRecord(offense_type=OffenseType.THEFT, complainant=Person(name="X"))


def test_default_penal_code_sections_filled():
    record = _minimal_record()
    assert record.penal_code_sections
    assert "৩৭৯" in record.penal_code_sections[0]


def test_explicit_penal_code_sections_preserved():
    record = _minimal_record(penal_code_sections=["কাস্টম ধারা"])
    assert record.penal_code_sections == ["কাস্টম ধারা"]


def test_with_defaults_filled_copies_complainant_to_victim():
    record = _minimal_record()
    filled = record.with_defaults_filled()
    assert filled.victim is not None
    assert filled.victim.name == record.complainant.name


def test_property_item_optional_fields():
    item = PropertyItem(description="মানিব্যাগ")
    assert item.quantity == 1
    assert item.estimated_value_bdt is None


def test_round_trip_json():
    record = _minimal_record(
        stolen_or_damaged_items=[PropertyItem(description="মোবাইল ফোন", estimated_value_bdt=10000)]
    )
    dumped = record.model_dump_json()
    restored = FIRRecord.model_validate_json(dumped)
    assert restored == record
