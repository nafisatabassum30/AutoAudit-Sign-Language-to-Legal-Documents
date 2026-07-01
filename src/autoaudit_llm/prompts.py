"""Prompt templates for the Bangla FIR LLM.

The model is trained/instructed to take an informal Bangla statement (the raw
output of the sign-language recognition stage) and produce a strict JSON object
matching :class:`autoaudit_llm.schema.FIRComplaint`. Emitting JSON (rather than
free text) makes the output machine-parseable and lets us validate/repair it
before rendering the final FIR document.
"""

from __future__ import annotations

import json

from .schema import FIRComplaint

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ আইন সহকারী, যিনি বাংলাদেশ পুলিশের প্রথম তথ্য বিবরণী (এফআইআর) "
    "প্রস্তুতে দক্ষ। ব্যবহারকারী ইশারা ভাষা (BdSL) থেকে রূপান্তরিত একটি অনানুষ্ঠানিক "
    "বাংলা বিবৃতি দেবেন। আপনার কাজ হলো সেই বিবৃতি থেকে ঘটনার তথ্য শনাক্ত করে একটি "
    "আনুষ্ঠানিক, আইনসম্মত এফআইআর তৈরি করা।\n\n"
    "নিয়মাবলী:\n"
    "১. শুধুমাত্র একটি বৈধ JSON অবজেক্ট আউটপুট দিন, অতিরিক্ত কোনো লেখা নয়।\n"
    "২. JSON-এ নিম্নলিখিত কী থাকবে: offense_type, incident_date, incident_time, "
    "location, complainant_name, victim_name, accused_name, stolen_items (তালিকা), "
    "complaint_body।\n"
    "৩. কোনো তথ্য বিবৃতিতে না থাকলে সেই ক্ষেত্রে \"অজ্ঞাত\" লিখুন (stolen_items হলে খালি তালিকা)।\n"
    "৪. complaint_body হবে আনুষ্ঠানিক আইনি বাংলায় লেখা সম্পূর্ণ অভিযোগের বিবরণ।\n"
    "৫. কোনো তথ্য বানিয়ে লিখবেন না; শুধু বিবৃতিতে থাকা তথ্য ব্যবহার করুন।"
)

INSTRUCTION = (
    "নিচের অনানুষ্ঠানিক বাংলা বিবৃতি থেকে একটি আনুষ্ঠানিক এফআইআর JSON তৈরি করুন।"
)

# Keys expected in the model's JSON output, kept in one place for validation.
FIR_JSON_KEYS = [
    "offense_type",
    "incident_date",
    "incident_time",
    "location",
    "complainant_name",
    "victim_name",
    "accused_name",
    "stolen_items",
    "complaint_body",
]


def build_user_prompt(raw_statement: str) -> str:
    """Compose the user turn shown to the model at train and inference time."""
    return f"{INSTRUCTION}\n\nবিবৃতি:\n\"\"\"\n{raw_statement.strip()}\n\"\"\""


def build_chat_messages(raw_statement: str) -> list[dict[str, str]]:
    """Return chat-format messages (system + user) for chat-template models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(raw_statement)},
    ]


def target_json(complaint: FIRComplaint) -> str:
    """Serialize a complaint to the exact JSON string used as the training target."""
    return json.dumps(complaint.model_dump(), ensure_ascii=False)


def build_training_text(raw_statement: str, complaint: FIRComplaint) -> str:
    """Render a full (prompt + completion) training example as plain text.

    This fallback format is used when a tokenizer chat template is not
    available. It mirrors a simple instruction-tuning layout.
    """
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{build_user_prompt(raw_statement)}\n"
        f"<|assistant|>\n{target_json(complaint)}"
    )
