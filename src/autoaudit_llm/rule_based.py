"""Deterministic, rule-based Bangla FIR builder.

This module provides a lightweight entity extractor + FIR composer that does
**not** require any model weights or GPU. It serves two purposes:

1. A dependable fallback for :class:`autoaudit_llm.inference.FIRGenerator` when
   the fine-tuned adapter is unavailable (no GPU / weights not downloaded), so
   the whole AutoAudit pipeline stays runnable and demonstrable.
2. A transparent baseline to compare the fine-tuned LLM against.

It uses keyword/pattern matching over known offense types, locations, times and
valuables (shared with the synthetic data generator).
"""

from __future__ import annotations

import re
from typing import List, Optional

from . import data_generation as dg
from .data_generation import _build_complaint_body, _extraction_to_complaint
from .schema import FIRComplaint, FIRExtraction

UNKNOWN = "অজ্ঞাত"

# Offense keyword -> canonical offense type.
_OFFENSE_KEYWORDS = [
    (["ছিনতাই", "ছিনিয়ে"], "ছিনতাই", "অজ্ঞাতনামা ছিনতাইকারী"),
    (["চুরি", "চুরে", "খোয়া"], "চুরি", UNKNOWN),
    (["মারধর", "মেরেছে", "হামলা", "মার", "আঘাত"], "মারধর / হামলা", None),
    (["হুমকি", "প্রাণনাশ", "ভয় দেখ"], "হুমকি প্রদান", None),
    (["প্রতারণা", "প্রতারণ", "ঠকিয়ে"], "প্রতারণা", None),
    (["ভাঙচুর", "ভেঙে"], "ভাঙচুর", None),
]

_TIME_RE = re.compile(
    r"(সকাল|দুপুর|বিকেল|বিকাল|সন্ধ্যা|রাত|ভোর|গভীর রাত)\s*[০-৯0-9]*\s*(টা|টার)?"
)


def _detect_offense(text: str) -> tuple[str, Optional[str]]:
    for keywords, offense, default_accused in _OFFENSE_KEYWORDS:
        if any(k in text for k in keywords):
            return offense, default_accused
    return UNKNOWN, None


def _detect_location(text: str) -> Optional[str]:
    for loc in dg.LOCATIONS:
        if loc in text:
            return loc
    return None


def _detect_time(text: str) -> Optional[str]:
    for t in dg.TIMES:
        if t in text:
            return t
    m = _TIME_RE.search(text)
    if m:
        return m.group(0).strip()
    return None


def _detect_items(text: str) -> List[str]:
    return [item for item in dg.VALUABLES if item in text]


def _detect_names(text: str) -> tuple[Optional[str], Optional[str]]:
    """Best-effort name detection using the known name vocabulary.

    Returns (accused_name, victim_name). Names appearing in the text are
    treated as the accused; the complainant/victim is usually the signer and
    therefore not named in the terse statement.
    """
    all_names = dg.MALE_NAMES + dg.FEMALE_NAMES
    found = [n for n in all_names if n in text]
    accused = found[0] if found else None
    return accused, None


def extract_entities(raw_statement: str) -> FIRExtraction:
    """Heuristically extract FIR entities from an informal Bangla statement."""
    text = raw_statement.strip()
    offense, default_accused = _detect_offense(text)
    accused_named, _ = _detect_names(text)
    accused = accused_named or default_accused or UNKNOWN

    return FIRExtraction(
        complainant_name=UNKNOWN,
        victim_name=UNKNOWN,
        accused_name=accused,
        offense_type=offense,
        location=_detect_location(text),
        incident_date=UNKNOWN,
        incident_time=_detect_time(text),
        stolen_items=_detect_items(text),
        description=text,
    )


def build_complaint(raw_statement: str) -> FIRComplaint:
    """Extract entities and compose a full FIR complaint deterministically."""
    ext = extract_entities(raw_statement)
    # Reuse the shared narrative builder for a consistent formal tone.
    complaint = _extraction_to_complaint(ext)
    # If nothing meaningful was extracted, still produce a valid document.
    if not complaint.complaint_body:
        complaint.complaint_body = _build_complaint_body(ext)
    return complaint
