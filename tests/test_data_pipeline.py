from __future__ import annotations

import json

from bdsllm.data import build_examples, prepare_dataset
from bdsllm.schema import IncidentFacts
from bdsllm.templates import render_legal_complaint


def test_render_legal_complaint_contains_required_sections() -> None:
    facts = IncidentFacts(
        recognized_text="আমার ফোন চুরি হয়েছে",
        complainant_name="রহিমা",
        incident_location="উত্তরা",
        offense_type="চুরি",
    )

    complaint = render_legal_complaint(facts)

    assert "বিষয়: চুরি সংক্রান্ত অভিযোগ।" in complaint
    assert "ঘটনার বিবরণ" in complaint
    assert "প্রার্থনা" in complaint
    assert "আমার ফোন চুরি হয়েছে" in complaint


def test_build_examples_uses_existing_complaint_when_present() -> None:
    examples = build_examples(
        [
            {
                "recognized_text": "ব্যাগ ছিনতাই হয়েছে",
                "complaint_text": "প্রস্তুত অভিযোগ",
                "video_id": "v1",
            }
        ]
    )

    assert len(examples) == 1
    assert examples[0].response == "প্রস্তুত অভিযোগ"
    assert examples[0].metadata["video_id"] == "v1"
    assert "ব্যাগ ছিনতাই হয়েছে" in examples[0].instruction


def test_prepare_dataset_writes_instruction_jsonl(tmp_path) -> None:
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "train.jsonl"
    input_path.write_text(
        json.dumps({"recognized_text": "মোবাইল চুরি", "offense_type": "চুরি"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    count = prepare_dataset(input_path, output_path)

    assert count == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "instruction" in payload
    assert "response" in payload
    assert "মোবাইল চুরি" in payload["response"]
