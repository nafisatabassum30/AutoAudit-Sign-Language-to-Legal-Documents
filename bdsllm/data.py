from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from bdsllm.prompts import build_instruction
from bdsllm.schema import ComplaintExample, IncidentFacts
from bdsllm.templates import render_legal_complaint


SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl"}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported dataset format: {path.suffix}. Use CSV, JSON, or JSONL.")

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        return records

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload["records"]
        raise ValueError("JSON input must be a list or an object with a 'records' list.")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_examples(records: Iterable[dict[str, Any]]) -> list[ComplaintExample]:
    examples: list[ComplaintExample] = []
    for index, record in enumerate(records):
        facts = IncidentFacts.from_mapping(record)
        response = str(record.get("complaint_text") or record.get("response") or "").strip()
        if not response:
            response = render_legal_complaint(facts)
        examples.append(
            ComplaintExample(
                instruction=build_instruction(facts),
                response=response,
                metadata={
                    "source_index": index,
                    "video_id": record.get("video_id", ""),
                    "template_id": record.get("template_id", ""),
                },
            )
        )
    return examples


def write_jsonl(examples: Iterable[ComplaintExample], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")


def prepare_dataset(input_path: str | Path, output_path: str | Path) -> int:
    records = load_records(input_path)
    examples = build_examples(records)
    write_jsonl(examples, output_path)
    return len(examples)
