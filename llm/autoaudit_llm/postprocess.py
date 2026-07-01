# -*- coding: utf-8 -*-
"""Turn raw LLM text output into a validated FIRComplaint and a final,
human-readable FIR document ready for submission.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional, Tuple

from pydantic import ValidationError

from .schema import FIRComplaint

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class FIRParseError(ValueError):
    """Raised when the LLM output cannot be parsed into a FIRComplaint."""


def extract_json_object(text: str) -> str:
    """Extract the first top-level-looking JSON object from arbitrary text.

    Handles the common cases of models wrapping JSON in ```json fences or
    adding stray prose before/after the object.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise FIRParseError(f"No JSON object found in model output: {text!r}")
    return match.group(0)


def parse_fir_json(text: str) -> FIRComplaint:
    """Parse and validate raw model output text into a FIRComplaint."""
    json_str = extract_json_object(text)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise FIRParseError(f"Invalid JSON in model output: {exc}") from exc
    try:
        return FIRComplaint.model_validate(data)
    except ValidationError as exc:
        raise FIRParseError(f"JSON does not match FIRComplaint schema: {exc}") from exc


def try_parse_fir_json(text: str) -> Tuple[Optional[FIRComplaint], Optional[str]]:
    """Non-raising variant of :func:`parse_fir_json`.

    Returns ``(complaint, None)`` on success or ``(None, error_message)``.
    """
    try:
        return parse_fir_json(text), None
    except FIRParseError as exc:
        return None, str(exc)


FIR_DOCUMENT_TEMPLATE = """\
প্রথম তথ্য প্রতিবেদন (FIR) - খসড়া
===================================
থানা: {thana}
প্রস্তুতের তারিখ: {generated_on}

১। অভিযোগকারীর তথ্য
------------------------
নাম: {complainant_name}
ঠিকানা: {complainant_address}
মোবাইল: {complainant_phone}

২। ভিকটিমের তথ্য
------------------------
ভিকটিমের নাম: {victim_name}

৩। ঘটনার বিবরণ
------------------------
অপরাধের ধরন: {offense_type}
ঘটনার তারিখ: {incident_date}
ঘটনার সময়: {incident_time}
ঘটনাস্থল: {incident_location}

৪। অভিযুক্তের তথ্য
------------------------
নাম: {accused_name}
বর্ণনা: {accused_description}

৫। ক্ষতিগ্রস্ত সম্পত্তি
------------------------
{property_description}

৬। সাক্ষী
------------------------
{witnesses}

৭। সম্পূর্ণ বিবরণ (আইনি ভাষায়)
------------------------
{narrative}

-----------------------------------
অভিযোগকারীর স্বাক্ষর: ______________________
"""


def _or_default(value: Optional[str], default: str = "উল্লেখ নেই") -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


def render_fir_document(complaint: FIRComplaint, generated_on: Optional[str] = None) -> str:
    """Render a validated FIRComplaint into the final submittable document."""
    witnesses = (
        "\n".join(f"- {w}" for w in complaint.witnesses)
        if complaint.witnesses
        else "কোনো সাক্ষীর তথ্য পাওয়া যায়নি"
    )
    offense_type = complaint.offense_type
    offense_type = offense_type.value if hasattr(offense_type, "value") else offense_type
    return FIR_DOCUMENT_TEMPLATE.format(
        thana=_or_default(complaint.thana, "উল্লেখ নেই (নিকটস্থ থানায় দাখিল করুন)"),
        generated_on=generated_on or date.today().isoformat(),
        complainant_name=_or_default(complaint.complainant_name),
        complainant_address=_or_default(complaint.complainant_address),
        complainant_phone=_or_default(complaint.complainant_phone),
        victim_name=_or_default(complaint.victim_name, _or_default(complaint.complainant_name)),
        offense_type=offense_type,
        incident_date=_or_default(complaint.incident_date),
        incident_time=_or_default(complaint.incident_time),
        incident_location=_or_default(complaint.incident_location),
        accused_name=_or_default(complaint.accused_name),
        accused_description=_or_default(complaint.accused_description),
        property_description=_or_default(complaint.property_description, "প্রযোজ্য নয়"),
        witnesses=witnesses,
        narrative=_or_default(complaint.narrative),
    )
