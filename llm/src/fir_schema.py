"""Schema, validation, and rendering utilities for the FIR (First
Information Report) documents produced by the fine-tuned LLM.

The LLM is trained to emit a JSON object following ``FIR_FIELDS`` for
every incident described in raw Bangla text. This module is the single
source of truth for that schema so that data preparation, training,
inference, and evaluation all agree on the same structure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

# Ordered list of fields the model must produce. Order matters: it is
# used both for rendering the JSON target during training and for
# rendering the final human-readable FIR document.
FIR_FIELDS = (
    "date_of_occurrence",
    "time_of_occurrence",
    "location",
    "offense_type",
    "complainant_name",
    "complainant_address",
    "complainant_contact",
    "accused",
    "description",
)

# Value used whenever information cannot be determined from the input.
UNKNOWN = "উল্লেখ নেই"


@dataclass
class FIRRecord:
    """A structured representation of a First Information Report."""

    date_of_occurrence: str = UNKNOWN
    time_of_occurrence: str = UNKNOWN
    location: str = UNKNOWN
    offense_type: str = UNKNOWN
    complainant_name: str = UNKNOWN
    complainant_address: str = UNKNOWN
    complainant_contact: str = UNKNOWN
    accused: str = "অজ্ঞাত"
    description: str = UNKNOWN

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FIRRecord":
        known = {f.name for f in fields(cls)}
        clean = {k: str(v).strip() for k, v in data.items() if k in known and v is not None}
        return cls(**clean)

    def to_dict(self) -> Dict[str, str]:
        return {name: getattr(self, name) for name in FIR_FIELDS}

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class FIRParseError(ValueError):
    """Raised when the model output cannot be parsed into a FIRRecord."""


def extract_json_block(text: str) -> str:
    """Extract the first top-level JSON object found in ``text``.

    Model generations sometimes include leading/trailing chatter or
    markdown code fences, so we look for the first balanced ``{...}``
    block instead of assuming the whole string is JSON.
    """

    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    start = text.find("{")
    if start == -1:
        raise FIRParseError(f"No JSON object found in model output: {text!r}")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise FIRParseError(f"Unbalanced JSON object in model output: {text!r}")


def parse_fir_output(text: str) -> FIRRecord:
    """Parse raw model output text into a validated :class:`FIRRecord`."""

    block = extract_json_block(text)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        raise FIRParseError(f"Invalid JSON produced by model: {exc}\nRaw block: {block!r}") from exc
    if not isinstance(data, dict):
        raise FIRParseError(f"Expected a JSON object, got: {type(data)}")
    return FIRRecord.from_dict(data)


FIR_DOCUMENT_TEMPLATE = """\
প্রাথমিক তথ্য বিবরণী (First Information Report)
==================================================

ঘটনার তারিখ            : {date_of_occurrence}
ঘটনার সময়              : {time_of_occurrence}
ঘটনাস্থল                : {location}
অপরাধের ধরন             : {offense_type}

অভিযোগকারীর নাম          : {complainant_name}
অভিযোগকারীর ঠিকানা       : {complainant_address}
যোগাযোগ নম্বর            : {complainant_contact}

অভিযুক্ত                : {accused}

ঘটনার বিস্তারিত বিবরণ:
{description}
"""


def render_fir_document(record: FIRRecord) -> str:
    """Render a :class:`FIRRecord` as a formatted, submission-ready
    Bangla FIR document."""

    return FIR_DOCUMENT_TEMPLATE.format(**record.to_dict())
