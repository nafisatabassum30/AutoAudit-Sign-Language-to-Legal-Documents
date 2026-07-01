import json

from src.fir_parser import (
    FIRParseError,
    parse_and_validate,
    parse_llm_output,
    render_document,
    validate_and_fill_defaults,
)

SAMPLE_FIR = {
    "thana": "উত্তরা",
    "district": "ঢাকা",
    "date_of_occurrence": "2026-06-30",
    "time_of_occurrence": "17:00",
    "date_of_report": "2026-07-01",
    "place_of_occurrence": "উত্তরা এলাকা",
    "complainant_name": "মোঃ রফিকুল ইসলাম",
    "complainant_address": "উত্তরা, ঢাকা",
    "complainant_phone": "01712345678",
    "victim_name": "মোঃ রফিকুল ইসলাম",
    "accused_name": "অজ্ঞাতনামা",
    "offense_type": "চুরি",
    "penal_code_sections": ["দণ্ডবিধি ১৮৬০ - ধারা ৩৭৯"],
    "items_involved": ["মানিব্যাগ"],
    "narrative_bn": "অভিযোগকারী জানান যে় তার মানিব্যাগ চুরি হয়েছে।",
}


def test_parse_llm_output_clean_json():
    text = json.dumps(SAMPLE_FIR, ensure_ascii=False)
    parsed = parse_llm_output(text)
    assert parsed["thana"] == "উত্তরা"


def test_parse_llm_output_with_code_fence_and_prose():
    text = f"এখানে ফলাফল:\n```json\n{json.dumps(SAMPLE_FIR, ensure_ascii=False)}\n```\nধন্যবাদ।"
    parsed = parse_llm_output(text)
    assert parsed["offense_type"] == "চুরি"


def test_parse_llm_output_trailing_comma_repair():
    broken = json.dumps(SAMPLE_FIR, ensure_ascii=False)
    broken = broken[:-1] + ',}'  # inject a trailing comma before the closing brace
    parsed = parse_llm_output(broken)
    assert parsed["accused_name"] == "অজ্ঞাতনামা"


def test_parse_llm_output_no_json_raises():
    try:
        parse_llm_output("এখানে কোনো জেসন নেই।")
    except FIRParseError:
        pass
    else:
        raise AssertionError("Expected FIRParseError")


def test_validate_and_fill_defaults_fills_missing_fields():
    partial = {"thana": "মিরপুর"}
    filled = validate_and_fill_defaults(partial)
    assert filled["thana"] == "মিরপুর"
    assert filled["complainant_name"] == "অজানা"
    assert filled["penal_code_sections"] == []
    assert filled["items_involved"] == []


def test_parse_and_validate_never_raises_on_garbage():
    result = parse_and_validate("সম্পূর্ণ অসংলগ্ন আউটপুট, কোনো জেসন নেই")
    assert result["complainant_name"] == "অজানা"
    assert isinstance(result["penal_code_sections"], list)


def test_render_document_contains_key_fields():
    doc = render_document(SAMPLE_FIR)
    assert "উত্তরা" in doc
    assert "চুরি" in doc
    assert "মানিব্যাগ" in doc
    assert "দণ্ডবিধি ১৮৬০ - ধারা ৩৭৯" in doc


def test_render_document_handles_missing_optional_lists():
    fir = dict(SAMPLE_FIR)
    fir["penal_code_sections"] = []
    fir["items_involved"] = []
    doc = render_document(fir)
    assert "প্রযোজ্য নয়" in doc
