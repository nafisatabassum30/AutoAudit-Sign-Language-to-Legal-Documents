"""Robustly parse LLM output into a structured FIR dict, and render that
dict into a formatted, human-readable Bangla FIR document.

The fine-tuned LLM is instructed (see ``prompts.py``) to always answer with a
single JSON object. In practice LLM output can still be malformed (extra
prose, trailing commas, missing keys, markdown code fences, etc.), so this
module defensively extracts/repairs JSON and fills in safe defaults for any
missing required field rather than failing the whole pipeline.
"""

from __future__ import annotations

import json
import re

REQUIRED_STRING_FIELDS = [
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
    "narrative_bn",
]

REQUIRED_LIST_FIELDS = ["penal_code_sections", "items_involved"]

UNKNOWN_PLACEHOLDER = "অজানা"

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class FIRParseError(ValueError):
    """Raised when no JSON object at all could be recovered from the text."""


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def _extract_json_blob(text: str) -> str:
    text = _strip_code_fences(text)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise FIRParseError("No JSON object found in LLM output.")
    return match.group(0)


def _try_loose_json_repair(blob: str) -> dict:
    """Attempt a couple of cheap repairs for common LLM JSON mistakes."""
    repaired = blob
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # trailing commas
    repaired = repaired.replace("'", '"')
    return json.loads(repaired)


def parse_llm_output(text: str) -> dict:
    """Extract a dict from raw LLM text, raising FIRParseError only if no
    JSON-like object can be located at all."""
    blob = _extract_json_blob(text)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return _try_loose_json_repair(blob)


def validate_and_fill_defaults(data: dict) -> dict:
    """Return a copy of ``data`` guaranteed to contain every required FIR
    field, filling missing/invalid ones with safe placeholders."""
    result = dict(data) if isinstance(data, dict) else {}

    for key in REQUIRED_STRING_FIELDS:
        value = result.get(key)
        if not isinstance(value, str) or not value.strip():
            result[key] = UNKNOWN_PLACEHOLDER

    for key in REQUIRED_LIST_FIELDS:
        value = result.get(key)
        if not isinstance(value, list):
            result[key] = []
        else:
            result[key] = [str(v) for v in value if str(v).strip()]

    return result


def parse_and_validate(text: str) -> dict:
    """Convenience wrapper: parse LLM text, falling back to an all-unknown
    record (rather than raising) if absolutely no JSON could be recovered.
    This keeps the downstream pipeline (and API) resilient."""
    try:
        data = parse_llm_output(text)
    except (FIRParseError, json.JSONDecodeError):
        data = {}
    return validate_and_fill_defaults(data)


_DOCUMENT_TEMPLATE = """\
বাংলাদেশ পুলিশ - প্রাথমিক অভিযোগ (FIR) ফরম
=============================================
থানা:                 {thana}
জেলা:                 {district}

ঘটনার তারিখ:           {date_of_occurrence}
ঘটনার সময়:            {time_of_occurrence}
অভিযোগ দায়েরের তারিখ:  {date_of_report}
ঘটনাস্থল:              {place_of_occurrence}

অভিযোগকারীর নাম:        {complainant_name}
অভিযোগকারীর ঠিকানা:     {complainant_address}
অভিযোগকারীর ফোন:        {complainant_phone}

ভিকটিমের নাম:           {victim_name}
অভিযুক্ত/সন্দেহভাজন:     {accused_name}

অপরাধের ধরন:           {offense_type}
প্রযোজ্য ধারা:          {sections_str}
সংশ্লিষ্ট মালামাল:       {items_str}

ঘটনার বিস্তারিত বিবরণ:
{narrative_bn}
=============================================
"""


def render_document(fir: dict) -> str:
    """Render a validated FIR dict into a formatted plain-text Bangla FIR
    document ready for review/printing/submission."""
    fir = validate_and_fill_defaults(fir)
    sections_str = "، ".join(fir["penal_code_sections"]) or "প্রযোজ্য নয়"
    items_str = "، ".join(fir["items_involved"]) or "প্রযোজ্য নয়"
    return _DOCUMENT_TEMPLATE.format(
        **{**fir, "sections_str": sections_str, "items_str": items_str}
    )
