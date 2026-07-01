"""Parse, repair and validate the LLM's JSON output into a :class:`FIRComplaint`.

LLMs occasionally wrap JSON in markdown fences, add trailing prose, or drop a
field. These helpers make the downstream contract robust by extracting the
first JSON object, filling missing keys with a sentinel and validating against
the pydantic schema.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .prompts import FIR_JSON_KEYS
from .schema import FIRComplaint

UNKNOWN = "অজ্ঞাত"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from arbitrary model text."""
    if not text:
        return None

    # Strip common markdown code fences.
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Fast path: the whole thing is JSON.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: greedily find a {...} span and try to parse progressively
    # shorter suffixes to tolerate trailing junk.
    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        return None
    span = match.group(0)
    for end in range(len(span), 1, -1):
        candidate = span[:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def coerce_to_complaint(data: Dict[str, Any]) -> FIRComplaint:
    """Fill missing/invalid fields and build a validated :class:`FIRComplaint`."""
    normalized: Dict[str, Any] = {}
    for key in FIR_JSON_KEYS:
        value = data.get(key)
        if key == "stolen_items":
            if isinstance(value, str):
                value = [v.strip() for v in re.split(r"[,、]", value) if v.strip()]
            elif not isinstance(value, list):
                value = []
            normalized[key] = value
        else:
            if value is None or (isinstance(value, str) and not value.strip()):
                value = UNKNOWN
            normalized[key] = str(value)
    return FIRComplaint(**normalized)


def parse_model_output(text: str) -> Optional[FIRComplaint]:
    """Full pipeline: text -> JSON -> validated complaint (or ``None``)."""
    obj = extract_json_object(text)
    if obj is None:
        return None
    try:
        return coerce_to_complaint(obj)
    except Exception:
        return None
