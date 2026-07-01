import json

import pytest

from src.postprocess import FIRParseError, parse_fir_output
from src.schema import FIRRecord, OffenseType, Person


def _valid_payload() -> dict:
    record = FIRRecord(
        offense_type=OffenseType.THEFT,
        complainant=Person(name="করিম উদ্দিন"),
        incident_location="উত্তরা",
        narrative_bn="একটি পরীক্ষামূলক বিবরণ।",
    )
    return json.loads(record.model_dump_json())


def test_parse_plain_json():
    payload = _valid_payload()
    raw = json.dumps(payload, ensure_ascii=False)
    result = parse_fir_output(raw)
    assert result.record.offense_type == OffenseType.THEFT
    assert not result.repaired


def test_parse_json_in_markdown_fence():
    payload = _valid_payload()
    raw = f"এখানে ফলাফল:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\nধন্যবাদ।"
    result = parse_fir_output(raw)
    assert result.record.incident_location == "উত্তরা"


def test_parse_json_with_trailing_comma_is_repaired():
    payload = _valid_payload()
    raw_dict_text = json.dumps(payload, ensure_ascii=False)
    # inject a trailing comma before the final closing brace
    broken = raw_dict_text[:-1] + ",}"
    result = parse_fir_output(broken)
    assert result.repaired is True
    assert result.record.offense_type == OffenseType.THEFT


def test_parse_injects_original_input_text():
    payload = _valid_payload()
    payload["raw_input_text"] = None
    raw = json.dumps(payload, ensure_ascii=False)
    result = parse_fir_output(raw, original_input_text="মূল ইনপুট টেক্সট")
    assert result.record.raw_input_text == "মূল ইনপুট টেক্সট"


def test_parse_no_json_raises():
    with pytest.raises(FIRParseError):
        parse_fir_output("এটি কোনো JSON নয়, শুধু একটি বাক্য।")


def test_parse_invalid_schema_raises():
    raw = json.dumps({"foo": "bar"})
    with pytest.raises(FIRParseError):
        parse_fir_output(raw)
