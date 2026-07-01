import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.fir_schema import (
    FIR_FIELDS,
    FIRParseError,
    FIRRecord,
    parse_fir_output,
    render_fir_document,
)


def test_from_dict_fills_all_fields():
    record = FIRRecord.from_dict({"date_of_occurrence": "০১/০৬/২০২৬", "location": "উত্তরা"})
    assert record.date_of_occurrence == "০১/০৬/২০২৬"
    assert record.location == "উত্তরা"
    # Unset fields fall back to defaults.
    assert record.offense_type == "উল্লেখ নেই"
    assert record.accused == "অজ্ঞাত"


def test_to_dict_has_all_expected_keys_in_order():
    record = FIRRecord()
    assert tuple(record.to_dict().keys()) == FIR_FIELDS


def test_round_trip_json():
    record = FIRRecord.from_dict({"location": "গুলশান", "offense_type": "চুরি"})
    parsed = parse_fir_output(record.to_json())
    assert parsed.location == "গুলশান"
    assert parsed.offense_type == "চুরি"


def test_parse_fir_output_strips_markdown_fence_and_chatter():
    raw = 'নিশ্চিত, এখানে JSON: \n```json\n{"location": "মিরপুর", "offense_type": "ছিনতাই"}\n```\nধন্যবাদ।'
    parsed = parse_fir_output(raw)
    assert parsed.location == "মিরপুর"
    assert parsed.offense_type == "ছিনতাই"


def test_parse_fir_output_raises_on_missing_json():
    with pytest.raises(FIRParseError):
        parse_fir_output("এখানে কোনো JSON নেই।")


def test_parse_fir_output_raises_on_invalid_json():
    with pytest.raises(FIRParseError):
        parse_fir_output("{not valid json,,,}")


def test_render_fir_document_contains_all_values():
    record = FIRRecord.from_dict(
        {
            "date_of_occurrence": "০১/০৬/২০২৬",
            "location": "উত্তরা",
            "offense_type": "চুরি",
            "description": "বিস্তারিত বিবরণ এখানে।",
        }
    )
    document = render_fir_document(record)
    assert "০১/০৬/২০২৬" in document
    assert "উত্তরা" in document
    assert "চুরি" in document
    assert "বিস্তারিত বিবরণ এখানে।" in document
