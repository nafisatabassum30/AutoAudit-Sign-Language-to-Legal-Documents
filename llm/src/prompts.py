"""Prompt construction for the Bangla-FIR fine-tuned LLM.

The upstream pipeline stage (ST-GNN sign recognition) produces short,
informal Bangla text such as::

    "আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকাল ৫টায়"

The LLM's job is to turn that into a structured JSON object matching
``fir_schema.FIR_FIELDS`` which downstream code renders into a formal
FIR document. This module defines the exact prompt format used both
at training time and at inference time -- they must match exactly.
"""

from __future__ import annotations

from typing import Dict, List

from .fir_schema import FIR_FIELDS

SYSTEM_PROMPT = (
    "You are a legal-drafting assistant for the Bangladesh Police. You receive short, "
    "informal Bangla sentences describing an incident, produced automatically from "
    "Bangladeshi Sign Language (BdSL) recognition. Your task is to convert that raw text "
    "into a single, valid JSON object describing a First Information Report (FIR), so it "
    "can be filed as a formal legal complaint.\n\n"
    "Rules:\n"
    "1. Output ONLY a single JSON object -- no extra commentary, no markdown fences.\n"
    "2. The JSON object must contain exactly these keys, in this order: "
    f"{', '.join(FIR_FIELDS)}.\n"
    "3. All values must be written in formal, grammatically correct Bangla.\n"
    "4. If a piece of information is missing from the input, use \"উল্লেখ নেই\" for that "
    "field (use \"অজ্ঞাত\" specifically for an unknown accused/suspect).\n"
    "5. The 'description' field must be a complete, formal paragraph (2-4 sentences) "
    "suitable for a legal complaint, expanding on the raw input -- do not merely repeat it."
)

USER_PROMPT_TEMPLATE = (
    "নিম্নলিখিত বাংলা বাক্যটি (সাংকেতিক ভাষা থেকে প্রাপ্ত) থেকে একটি FIR JSON তৈরি করুন:\n\n"
    "ইনপুট: {input_text}"
)


def build_chat_messages(input_text: str) -> List[Dict[str, str]]:
    """Return a chat-formatted message list for the given raw input text."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(input_text=input_text.strip())},
    ]


def build_prompt(tokenizer, input_text: str) -> str:
    """Render the full prompt string using the tokenizer's chat template.

    Falls back to a plain-text template if the tokenizer has no chat
    template configured (e.g. base, non-instruct checkpoints).
    """

    messages = build_chat_messages(input_text)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### User:\n{messages[1]['content']}\n\n"
        f"### Response:\n"
    )
