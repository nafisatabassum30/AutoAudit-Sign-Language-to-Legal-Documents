from autoaudit_llm.postprocess import (
    coerce_to_complaint,
    extract_json_object,
    parse_model_output,
)


def test_extract_plain_json():
    obj = extract_json_object('{"a": 1, "b": "x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_json_with_markdown_fence():
    text = "```json\n{\"offense_type\": \"চুরি\"}\n```"
    obj = extract_json_object(text)
    assert obj == {"offense_type": "চুরি"}


def test_extract_json_with_trailing_prose():
    text = 'Here is the FIR: {"offense_type": "চুরি"} thanks!'
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["offense_type"] == "চুরি"


def test_extract_returns_none_for_garbage():
    assert extract_json_object("no json here") is None


def test_coerce_fills_missing_fields():
    complaint = coerce_to_complaint({"offense_type": "চুরি"})
    assert complaint.offense_type == "চুরি"
    assert complaint.incident_date == "অজ্ঞাত"
    assert complaint.stolen_items == []


def test_coerce_splits_string_items():
    complaint = coerce_to_complaint({"stolen_items": "মোবাইল, মানিব্যাগ"})
    assert complaint.stolen_items == ["মোবাইল", "মানিব্যাগ"]


def test_parse_model_output_end_to_end():
    text = '{"offense_type": "ছিনতাই", "complaint_body": "বিবরণ"}'
    complaint = parse_model_output(text)
    assert complaint is not None
    assert complaint.offense_type == "ছিনতাই"


def test_parse_model_output_none_on_failure():
    assert parse_model_output("totally not json") is None
