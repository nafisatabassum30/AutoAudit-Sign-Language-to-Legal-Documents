#!/usr/bin/env python3
"""Flask API exposing the LLM stage: raw Bangla sign-recognition text in,
structured FIR JSON + formatted legal document out.

This is the "bridge" service between the upstream ST-GNN sign-recognition
stage and downstream FIR submission/review tooling described in the
AutoAudit pipeline.

Endpoints:
    GET  /health
    POST /api/v1/generate-fir
        body: {"text": "<raw bangla text>", "complainant_name": "...",
                "complainant_address": "...", "complainant_phone": "..."}
        -> {"fir": {...}, "document": "...", "llm_raw_output": "..."}

Configuration via environment variables:
    FIR_BASE_MODEL   (default: Qwen/Qwen2.5-0.5B-Instruct)
    FIR_ADAPTER_PATH (default: unset -> base model only, no fine-tuning)
    FIR_MAX_NEW_TOKENS (default: 512)

Run:
    FIR_ADAPTER_PATH=checkpoints/fir-lora/final python api/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.infer import FIRGenerator  # noqa: E402

app = Flask(__name__)

_generator: FIRGenerator | None = None


def get_generator() -> FIRGenerator:
    global _generator
    if _generator is None:
        base_model = os.environ.get("FIR_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        adapter_path = os.environ.get("FIR_ADAPTER_PATH") or None
        max_new_tokens = int(os.environ.get("FIR_MAX_NEW_TOKENS", "512"))
        app.logger.info("Loading FIR generator: base_model=%s adapter=%s", base_model, adapter_path)
        _generator = FIRGenerator(base_model, adapter_path, max_new_tokens=max_new_tokens)
    return _generator


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _generator is not None})


@app.post("/api/v1/generate-fir")
def generate_fir():
    payload = request.get_json(silent=True) or {}
    raw_text = (payload.get("text") or "").strip()
    if not raw_text:
        return jsonify({"error": "Field 'text' (raw signed text) is required."}), 400

    try:
        result = get_generator().generate_fir(raw_text)
    except Exception as exc:  # noqa: BLE001 - surface model/runtime errors to the caller
        app.logger.exception("FIR generation failed")
        return jsonify({"error": f"Generation failed: {exc}"}), 500

    fir = result["fir"]
    # Allow the caller (e.g. a mobile app that knows the logged-in user) to
    # override complainant identity fields the LLM couldn't infer from the
    # short signed utterance alone.
    for field in ("complainant_name", "complainant_address", "complainant_phone"):
        if payload.get(field):
            fir[field] = payload[field]

    return jsonify(
        {
            "fir": fir,
            "document": result["document"],
            "llm_raw_output": result["llm_raw_output"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
