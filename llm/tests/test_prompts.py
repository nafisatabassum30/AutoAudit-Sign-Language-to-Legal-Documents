import json

from src.prompts import build_chat_messages, build_plain_prompt, build_training_example


def test_build_chat_messages_has_system_and_user():
    messages = build_chat_messages("আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "উত্তরায়" in messages[1]["content"]


def test_build_training_example_appends_assistant_json():
    output_json = {"offense_type": "চুরি", "incident_location": "উত্তরা"}
    messages = build_training_example("ইনপুট টেক্সট", output_json)
    assert messages[-1]["role"] == "assistant"
    parsed = json.loads(messages[-1]["content"])
    assert parsed == output_json


def test_build_plain_prompt_contains_input():
    prompt = build_plain_prompt("পরীক্ষামূলক ইনপুট")
    assert "পরীক্ষামূলক ইনপুট" in prompt
    assert "আউটপুট" in prompt
