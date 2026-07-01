from autoaudit_llm.rule_based import build_complaint, extract_entities


def test_detects_theft_location_time_item():
    ext = extract_entities("আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকেল ৫টা")
    assert ext.offense_type == "চুরি"
    assert ext.location == "উত্তরা"
    assert ext.incident_time == "বিকেল ৫টা"
    assert "মানিব্যাগ" in ext.stolen_items


def test_detects_mugging():
    ext = extract_entities("মিরপুর ১০ রাত ৯টা আমার মোবাইল ফোন ছিনতাই হয়েছে")
    assert ext.offense_type == "ছিনতাই"
    assert ext.location == "মিরপুর ১০"
    assert "মোবাইল ফোন" in ext.stolen_items


def test_detects_assault_with_named_accused():
    ext = extract_entities("করিম মিয়া আমাকে মারধর করেছে ধানমন্ডি ২৭ সন্ধ্যা ৬টা")
    assert ext.offense_type == "মারধর / হামলা"
    assert ext.accused_name == "করিম মিয়া"


def test_build_complaint_produces_valid_document():
    complaint = build_complaint("আমার মোবাইল ফোন চুরি হয়েছে গুলশান ১ দুপুর ২টা")
    assert complaint.offense_type == "চুরি"
    doc = complaint.to_document()
    assert "প্রথম তথ্য বিবরণী" in doc
    assert complaint.complaint_body
