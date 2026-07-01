from src.prompts import (
    REQUIRED_FIR_KEYS,
    build_chat_messages,
    build_training_example,
    build_user_prompt,
    render_prompt_plain,
)


def test_build_user_prompt_includes_raw_text():
    prompt = build_user_prompt("আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা")
    assert "মানিব্যাগ" in prompt


def test_build_chat_messages_has_system_and_user():
    messages = build_chat_messages("আমার মানিব্যাগ চুরি")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert all(key in messages[0]["content"] for key in REQUIRED_FIR_KEYS)


def test_build_training_example_appends_assistant_json():
    fir = {"thana": "উত্তরা"}
    messages = build_training_example("আমার মানিব্যাগ চুরি", fir)
    assert messages[-1]["role"] == "assistant"
    assert "উত্তরা" in messages[-1]["content"]


def test_render_prompt_plain_is_a_single_string():
    text = render_prompt_plain("আমার মানিব্যাগ চুরি")
    assert isinstance(text, str)
    assert "মানিব্যাগ" in text
