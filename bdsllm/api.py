from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI

from bdsllm.inference import BanglaLegalComplaintGenerator
from bdsllm.schema import IncidentFacts


app = FastAPI(title="AutoAudit Bangla Legal LLM", version="0.1.0")


@lru_cache(maxsize=1)
def get_generator() -> BanglaLegalComplaintGenerator:
    return BanglaLegalComplaintGenerator(
        model_name_or_path=os.getenv("BANGLA_LEGAL_LLM_MODEL"),
        adapter_path=os.getenv("BANGLA_LEGAL_LLM_ADAPTER"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-complaint")
def generate_complaint(payload: dict[str, str]) -> dict[str, str]:
    facts = IncidentFacts.from_mapping(payload)
    complaint = get_generator().generate(facts)
    return {"complaint_text": complaint}
