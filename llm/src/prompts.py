"""Prompt construction for the BdSL -> FIR legal-document LLM stage.

The upstream ST-GNN sign-recognition stage emits short, telegraphic Bangla
text (e.g. "আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা"). This module builds the
chat prompt that asks the fine-tuned LLM to turn that raw text into a single
structured JSON object describing a First Information Report (FIR), which is
later rendered into a formal Bangla legal document by
``src/fir_parser.py::render_document``.
"""

from __future__ import annotations

import json

REQUIRED_FIR_KEYS = [
    "thana",
    "district",
    "date_of_occurrence",
    "time_of_occurrence",
    "date_of_report",
    "place_of_occurrence",
    "complainant_name",
    "complainant_address",
    "complainant_phone",
    "victim_name",
    "accused_name",
    "offense_type",
    "penal_code_sections",
    "items_involved",
    "narrative_bn",
]

SYSTEM_PROMPT = (
    "তুমি একজন বাংলাদেশ পুলিশের FIR (First Information Report) সহায়ক ভাষা মডেল। "
    "তোমাকে বাংলাদেশি সাংকেতিক ভাষা (BdSL) স্বীকৃতি ব্যবস্থা থেকে প্রাপ্ত সংক্ষিপ্ত, "
    "আনুষ্ঠানিকতাবিহীন বাংলা লেখা দেওয়া হবে, যা একজন বধির অভিযোগকারীর ইঙ্গিত থেকে "
    "স্বয়ংক্রিয়ভাবে তৈরি হয়েছে। তোমার কাজ হলো ঐ লেখা থেকে ঘটনার প্রয়োজনীয় তথ্য বুঝে "
    "নিয়ে একটি সম্পূর্ণ, আইনগতভাবে গ্রহণযোগ্য এফআইআর (FIR) তথ্যছক তৈরি করা।\n\n"
    "নিয়মাবলী:\n"
    "১. উত্তর অবশ্যই একটি একক বৈধ JSON অবজেক্ট হতে হবে, অন্য কোনো ব্যাখ্যা বা টেক্সট থাকবে না।\n"
    "২. নিচের সবগুলো key অবশ্যই থাকতে হবে: "
    + ", ".join(REQUIRED_FIR_KEYS)
    + "।\n"
    "৩. যদি ইনপুট থেকে কোনো তথ্য নির্ধারণ করা সম্ভব না হয়, উপযুক্ত ক্ষেত্রে "
    "\"অজ্ঞাতনামা\" (ব্যক্তির জন্য) বা \"অজানা\" (অন্য তথ্যের জন্য) ব্যবহার করো, "
    "কখনো তথ্য বানিয়ে বলবে না।\n"
    "৪. `narrative_bn` ক্ষেত্রে ৩-৫ বাক্যের একটি আনুষ্ঠানিক, নিরপেক্ষ ও আইনসম্মত বাংলা "
    "বিবরণ লিখতে হবে যা ঘটনার সময়, স্থান, সংশ্লিষ্ট ব্যক্তি ও ঘটনার প্রকৃতি বর্ণনা করে।\n"
    "৫. `penal_code_sections` ও `items_involved` অবশ্যই তালিকা (list) আকারে থাকতে হবে, "
    "প্রযোজ্য না হলে খালি তালিকা [] ব্যবহার করো।"
)


def build_user_prompt(raw_signed_text: str) -> str:
    """Wrap the raw sign-recognition text in a short user instruction."""
    return (
        "নিচের বাক্যাংশটি সাংকেতিক ভাষা থেকে স্বীকৃত রাও বাংলা টেক্সট। এটি বিশ্লেষণ করে "
        "সম্পূর্ণ FIR JSON তথ্যছক তৈরি করো।\n\n"
        f"রাও টেক্সট: {raw_signed_text.strip()}"
    )


def build_chat_messages(raw_signed_text: str) -> list[dict[str, str]]:
    """Return chat-format messages (system/user) for instruct-tuned models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(raw_signed_text)},
    ]


def build_training_example(raw_signed_text: str, fir_json: dict) -> list[dict[str, str]]:
    """Return full chat messages including the assistant target, for SFT."""
    messages = build_chat_messages(raw_signed_text)
    messages.append(
        {"role": "assistant", "content": json.dumps(fir_json, ensure_ascii=False, default=str)}
    )
    return messages


def render_prompt_plain(raw_signed_text: str) -> str:
    """Fallback plain-text prompt for base models without a chat template."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{build_user_prompt(raw_signed_text)}\n\n"
        "উত্তর (শুধুমাত্র JSON):\n"
    )
