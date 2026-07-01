import json

from prepare_dataset import split, validate_and_format

VALID_ROW = {
    "raw_text": "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫",
    "target": {
        "complainant_name": "মোঃ রফিকুল ইসলাম",
        "incident_location": "উত্তরা, ঢাকা",
        "offense_type": "চুরি",
        "accused_name": "অজ্ঞাত",
        "narrative": "অদ্য উত্তরা এলাকায় অভিযোগকারীর ওয়ালেট চুরি হয়েছে।",
    },
}

INVALID_ROW = {
    "raw_text": "কিছু একটা ঘটেছে",
    "target": {
        "incident_location": "ঢাকা",
        "offense_type": "এটি বৈধ অপরাধের ধরন নয়",
        "narrative": "বিবরণ",
    },
}


def test_validate_and_format_keeps_valid_and_drops_invalid():
    formatted = validate_and_format([VALID_ROW, INVALID_ROW])
    assert len(formatted) == 1
    assert formatted[0]["raw_text"] == VALID_ROW["raw_text"]


def test_formatted_rows_have_three_chat_turns_with_assistant_json():
    formatted = validate_and_format([VALID_ROW])
    messages = formatted[0]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant"]
    parsed_back = json.loads(messages[-1]["content"])
    assert parsed_back["offense_type"] == "চুরি"


def test_split_respects_fractions_and_covers_all_rows():
    rows = [{"i": i} for i in range(20)]
    train, val, test = split(rows, seed=0, val_frac=0.2, test_frac=0.2)
    assert len(train) + len(val) + len(test) == 20
    assert len(val) == 4
    assert len(test) == 4
    all_i = {r["i"] for r in (train + val + test)}
    assert all_i == set(range(20))
