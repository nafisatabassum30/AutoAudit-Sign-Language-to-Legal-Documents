import pytest
from pydantic import ValidationError

from autoaudit_llm.schema import ALL_KEYS, FIRComplaint, OffenseType


def test_minimal_valid_complaint():
    complaint = FIRComplaint(
        incident_location="উত্তরা, ঢাকা",
        offense_type="চুরি",
        narrative="অদ্য উত্তরা এলাকায় অভিযোগকারীর ওয়ালেট চুরি হয়েছে।",
    )
    assert complaint.offense_type == OffenseType.THEFT.value
    assert complaint.complainant_name == "অজ্ঞাত"


def test_all_offense_types_accepted():
    for offense in OffenseType:
        complaint = FIRComplaint(
            incident_location="ঢাকা",
            offense_type=offense.value,
            narrative="বিবরণ",
        )
        assert complaint.offense_type == offense.value


def test_invalid_offense_type_rejected():
    with pytest.raises(ValidationError):
        FIRComplaint(
            incident_location="ঢাকা",
            offense_type="not-a-real-offense",
            narrative="বিবরণ",
        )


def test_narrative_is_required():
    with pytest.raises(ValidationError):
        FIRComplaint(incident_location="ঢাকা", offense_type="চুরি")


def test_all_keys_matches_model_fields():
    assert set(ALL_KEYS) == set(FIRComplaint.model_fields.keys())
    assert "narrative" in ALL_KEYS
    assert "offense_type" in ALL_KEYS
