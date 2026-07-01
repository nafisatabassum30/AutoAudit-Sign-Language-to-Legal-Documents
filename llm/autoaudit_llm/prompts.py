# -*- coding: utf-8 -*-
"""Prompt construction for the FIR-generation LLM.

The model is instruction-tuned as a text -> JSON extractor/composer: given
raw, often telegraphic Bangla text (the output of the upstream BdSL
sign-recognition stage), it must emit *one* JSON object that matches
``autoaudit_llm.schema.FIRComplaint``.
"""
from __future__ import annotations

import json
from typing import Optional

from .schema import ALL_KEYS, OffenseType

SYSTEM_PROMPT = (
    "তুমি একজন বাংলাদেশ পুলিশের জন্য কাজ করা আইনি সহকারী AI। তোমার ইনপুট আসে "
    "একটি বধির/শ্রবণ-প্রতিবন্ধী ব্যবহারকারীর সাংকেতিক ভাষা (Bangladeshi Sign "
    "Language) থেকে স্বয়ংক্রিয়ভাবে চেনা শব্দের সংক্ষিপ্ত, অনানুষ্ঠানিক বাংলা বাক্য। "
    "তোমার কাজ হলো এই তথ্য থেকে একটি সম্পূর্ণ, আইনগতভাবে যথাযথ First Information "
    "Report (FIR) তৈরি করার জন্য একটিমাত্র বৈধ JSON অবজেক্ট আউটপুট করা। "
    "নিয়মাবলী: (1) কেবলমাত্র JSON আউটপুট দিবে, অন্য কোনো ব্যাখ্যা নয়। "
    "(2) নির্দিষ্ট তথ্য (নাম, তারিখ, সময়, স্থান) না পাওয়া গেলে সেই ফিল্ডে "
    "'অজ্ঞাত' বা 'উল্লেখ নেই' লিখবে, কল্পনা করে তথ্য বানাবে না। "
    "(3) 'narrative' ফিল্ডে ঘটনাটি সম্পূর্ণ, আনুষ্ঠানিক আইনি বাংলা ভাষায়, প্রথম "
    "পুরুষে ('অদ্য...', 'উক্ত সময়ে...') লিখবে। "
    "(4) 'offense_type' অবশ্যই নিচের তালিকার একটি মান হতে হবে: "
    + ", ".join(o.value for o in OffenseType)
)

INSTRUCTION = (
    "নিচের ইশারা ভাষা থেকে প্রাপ্ত বাংলা বিবরণ থেকে একটি FIR তৈরির জন্য প্রয়োজনীয় "
    "সব তথ্য বের করে JSON আউটপুট দাও। JSON-এর keys হতে হবে ঠিক এই "
    f"{len(ALL_KEYS)}টি: {json.dumps(ALL_KEYS, ensure_ascii=False)}"
)


def build_user_prompt(raw_bangla_text: str) -> str:
    return f"{INSTRUCTION}\n\nইশারা ভাষা থেকে প্রাপ্ত বিবরণ:\n\"{raw_bangla_text.strip()}\""


def build_chat_messages(raw_bangla_text: str, assistant_json: Optional[str] = None):
    """Build a chat-formatted message list usable with any HF chat template.

    If ``assistant_json`` is provided, an assistant turn is appended too, so
    the same helper can build both training examples and inference prompts.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(raw_bangla_text)},
    ]
    if assistant_json is not None:
        messages.append({"role": "assistant", "content": assistant_json})
    return messages
