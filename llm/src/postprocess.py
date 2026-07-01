"""Robust parsing of raw LLM output text into a validated :class:`FIRRecord`.

LLMs occasionally wrap JSON in markdown fences, add a stray trailing comma,
or emit a short explanatory sentence before/after the JSON object. This
module extracts the JSON, applies a handful of cheap deterministic repairs,
and validates against the schema -- raising a clear, actionable error if the
generation truly cannot be salvaged.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from .schema import FIRRecord

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class FIRParseError(ValueError):
    """Raised when raw model output cannot be parsed into a valid FIRRecord."""


@dataclass
class ParseResult:
    record: FIRRecord
    raw_json_text: str
    repaired: bool


def _extract_json_blob(text: str) -> str:
    fence_match = _CODE_FENCE_RE.search(text)
    candidate = fence_match.group(1) if fence_match else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FIRParseError(f"No JSON object found in model output: {text!r}")
    return candidate[start : end + 1]


def _repair_json_text(blob: str) -> str:
    repaired = _TRAILING_COMMA_RE.sub(r"\1", blob)
    return repaired


def parse_fir_output(raw_text: str, *, original_input_text: Optional[str] = None) -> ParseResult:
    """Parse ``raw_text`` (raw LLM generation) into a validated FIRRecord.

    Raises :class:`FIRParseError` if parsing/validation ultimately fails.
    """
    blob = _extract_json_blob(raw_text)

    repaired = False
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        blob_fixed = _repair_json_text(blob)
        try:
            data = json.loads(blob_fixed)
            repaired = True
        except json.JSONDecodeError as e:
            raise FIRParseError(f"Could not decode JSON even after repair: {e}\nBlob: {blob!r}") from e

    if original_input_text and not data.get("raw_input_text"):
        data["raw_input_text"] = original_input_text

    try:
        record = FIRRecord.model_validate(data)
    except ValidationError as e:
        raise FIRParseError(f"Schema validation failed: {e}") from e

    return ParseResult(record=record, raw_json_text=blob, repaired=repaired)
