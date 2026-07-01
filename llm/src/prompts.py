"""Prompt / chat-template construction shared by training and inference.

Keeping this in one place guarantees the exact same prompt format is used
when we fine-tune the model and when we later call it for inference --
a very common source of silent quality regressions otherwise.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .schema import FIRRecord

SYSTEM_PROMPT = (
    "তুমি একজন আইনি সহকারী AI, যা বাংলাদেশ সাইন ল্যাংগুয়েজ (BdSL) থেকে "
    "স্বয়ংক্রিয়ভাবে চিহ্নিত অস্পষ্ট/অনানুষ্ঠানিক বাংলা বাক্য পড়ে সেটিকে একটি "
    "First Information Report (FIR)-উপযোগী স্ট্রাকচার্ড JSON-এ রূপান্তর করো।\n\n"
    "নিয়মাবলী:\n"
    "1. আউটপুট অবশ্যই একটিমাত্র বৈধ (valid) JSON object হতে হবে, অন্য কোনো "
    "টেক্সট, ব্যাখ্যা, বা মার্কডাউন কোড-ফেন্স ছাড়া।\n"
    "2. JSON-এর keys নিচের schema অনুযায়ী হতে হবে: offense_type, "
    "penal_code_sections, complainant, victim, accused, accused_unknown, "
    "incident_date, incident_time, incident_location, police_station, "
    "district, stolen_or_damaged_items, witnesses, narrative_bn, "
    "raw_input_text।\n"
    "3. তথ্য না থাকলে সংশ্লিষ্ট ক্ষেত্র null অথবা খালি list রাখো, কখনো তথ্য "
    "বানিয়ে দিও না (hallucinate করো না)।\n"
    "4. narrative_bn অবশ্যই আনুষ্ঠানিক, আইনি-উপযোগী, ব্যাকরণগতভাবে শুদ্ধ বাংলায় "
    "লিখতে হবে, ইনপুট টেক্সট নিজে কপি করলে চলবে না।\n"
    "5. raw_input_text ক্ষেত্রে মূল ইনপুট টেক্সট অপরিবর্তিত রাখো।"
)

FEW_SHOT_INSTRUCTION_TEMPLATE = "স্বাক্ষর ভাষা থেকে প্রাপ্ত টেক্সট:\n{input_text}"


def build_chat_messages(input_text: str) -> List[Dict[str, str]]:
    """Build chat-formatted messages (system/user) for inference or training.

    Compatible with ``tokenizer.apply_chat_template`` for any instruct model
    (Llama-3, Qwen2.5, Gemma-2, TituLM, etc.).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_INSTRUCTION_TEMPLATE.format(input_text=input_text)},
    ]


def build_training_example(input_text: str, output_json: dict) -> List[Dict[str, str]]:
    """Full messages list including the target assistant turn, for SFT."""
    messages = build_chat_messages(input_text)
    assistant_content = json.dumps(output_json, ensure_ascii=False)
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def build_plain_prompt(input_text: str) -> str:
    """Fallback non-chat-template prompt (base/completion-style models)."""
    return (
        f"### নির্দেশনা:\n{SYSTEM_PROMPT}\n\n"
        f"### ইনপুট:\n{input_text}\n\n"
        f"### আউটপুট (JSON):\n"
    )


def build_plain_training_text(input_text: str, output_json: dict) -> str:
    prompt = build_plain_prompt(input_text)
    target = json.dumps(output_json, ensure_ascii=False)
    return prompt + target
