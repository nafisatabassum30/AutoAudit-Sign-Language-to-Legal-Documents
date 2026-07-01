from autoaudit_llm.config import AppConfig, load_config
from autoaudit_llm.prompts import (
    FIR_JSON_KEYS,
    build_chat_messages,
    build_training_text,
    target_json,
)
from autoaudit_llm.schema import FIRComplaint


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.model.base_model
    assert cfg.lora.r > 0
    assert 0.0 <= cfg.data.eval_split < 1.0


def test_load_missing_config_returns_defaults(tmp_path):
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.api.port == 8000


def test_chat_messages_structure():
    messages = build_chat_messages("আমার মোবাইল চুরি হয়েছে")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "আমার মোবাইল চুরি হয়েছে" in messages[1]["content"]


def test_target_json_contains_keys():
    complaint = FIRComplaint(
        offense_type="চুরি", incident_date="১ মে ২০২৫", incident_time="বিকেল ৫টা",
        location="উত্তরা", complainant_name="ক", victim_name="ক",
        accused_name="অজ্ঞাত", stolen_items=["মোবাইল"], complaint_body="বিবরণ",
    )
    js = target_json(complaint)
    for key in FIR_JSON_KEYS:
        assert key in js


def test_build_training_text_has_all_roles():
    complaint = FIRComplaint(
        offense_type="চুরি", incident_date="অজ্ঞাত", incident_time="অজ্ঞাত",
        location="অজ্ঞাত", complainant_name="অজ্ঞাত", victim_name="অজ্ঞাত",
        accused_name="অজ্ঞাত", stolen_items=[], complaint_body="বিবরণ",
    )
    text = build_training_text("বিবৃতি", complaint)
    assert "<|system|>" in text and "<|user|>" in text and "<|assistant|>" in text
