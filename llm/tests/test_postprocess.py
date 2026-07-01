import json

from autoaudit_llm.postprocess import (
    FIRParseError,
    extract_json_object,
    parse_fir_json,
    render_fir_document,
    try_parse_fir_json,
)

VALID_TARGET = {
    "complainant_name": "মোঃ রফিকুল ইসলাম",
    "complainant_address": None,
    "complainant_phone": None,
    "victim_name": None,
    "incident_date": "28 June 2026",
    "incident_time": "বিকেল ৫টা",
    "incident_location": "উত্তরা, ঢাকা",
    "thana": "উত্তরা পশ্চিম থানা",
    "offense_type": "চুরি",
    "accused_name": "অজ্ঞাত",
    "accused_description": None,
    "property_description": "একটি ওয়ালেট",
    "witnesses": [],
    "narrative": "অদ্য উত্তরা এলাকায় অভিযোগকারীর ওয়ালেট চুরি হয়েছে।",
}


def test_extract_json_object_from_fenced_block():
    text = f"এখানে ফলাফল:\n```json\n{json.dumps(VALID_TARGET, ensure_ascii=False)}\n```\nধন্যবাদ।"
    extracted = json.loads(extract_json_object(text))
    assert extracted["offense_type"] == "চুরি"


def test_extract_json_object_bare():
    text = json.dumps(VALID_TARGET, ensure_ascii=False)
    extracted = json.loads(extract_json_object(text))
    assert extracted["complainant_name"] == "মোঃ রফিকুল ইসলাম"


def test_parse_fir_json_success():
    text = json.dumps(VALID_TARGET, ensure_ascii=False)
    complaint = parse_fir_json(text)
    assert complaint.incident_location == "উত্তরা, ঢাকা"


def test_parse_fir_json_invalid_raises():
    try:
        parse_fir_json("this is not json at all")
        assert False, "expected FIRParseError"
    except FIRParseError:
        pass


def test_try_parse_fir_json_graceful_failure():
    complaint, error = try_parse_fir_json("not json")
    assert complaint is None
    assert error is not None


def test_render_fir_document_contains_key_fields():
    complaint, error = try_parse_fir_json(json.dumps(VALID_TARGET, ensure_ascii=False))
    assert error is None
    doc = render_fir_document(complaint)
    assert "প্রথম তথ্য প্রতিবেদন" in doc
    assert "মোঃ রফিকুল ইসলাম" in doc
    assert "চুরি" in doc
    assert "উত্তরা, ঢাকা" in doc


def test_render_fir_document_fills_defaults_for_missing_fields():
    minimal = {
        "incident_location": "ঢাকা",
        "offense_type": "অন্যান্য",
        "narrative": "বিবরণ পাওয়া যায়নি।",
    }
    complaint, error = try_parse_fir_json(json.dumps(minimal, ensure_ascii=False))
    assert error is None
    doc = render_fir_document(complaint)
    assert "উল্লেখ নেই" in doc
    assert "কোনো সাক্ষীর তথ্য পাওয়া যায়নি" in doc
