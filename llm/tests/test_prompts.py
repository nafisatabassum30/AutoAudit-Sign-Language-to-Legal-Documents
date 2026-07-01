import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fir_schema import FIR_FIELDS
from src.prompts import SYSTEM_PROMPT, build_chat_messages, build_prompt


def test_build_chat_messages_has_system_and_user_roles():
    messages = build_chat_messages("আমার মানিব্যাগ চুরি হয়েছে")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    assert "আমার মানিব্যাগ চুরি হয়েছে" in messages[1]["content"]


def test_system_prompt_mentions_all_fir_fields():
    for field in FIR_FIELDS:
        assert field in SYSTEM_PROMPT


class _FakeTokenizerNoChatTemplate:
    chat_template = None


def test_build_prompt_falls_back_without_chat_template():
    prompt = build_prompt(_FakeTokenizerNoChatTemplate(), "টেস্ট ইনপুট")
    assert "টেস্ট ইনপুট" in prompt
    assert "### Response:" in prompt
