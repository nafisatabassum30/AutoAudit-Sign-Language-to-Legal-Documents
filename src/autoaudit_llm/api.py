"""Flask API exposing the Bangla FIR LLM.

Endpoints
---------
GET  /health              -> service + model status
POST /generate            -> {"raw_statement": "..."} -> FIR complaint (JSON + rendered doc)
POST /extract             -> {"raw_statement": "..."} -> rule-based extracted entities

This is the ``Build Flask API to connect stages`` step of the pipeline: the
sign-recognition stage posts recognized Bangla text and receives a FIR-ready
complaint back.
"""

from __future__ import annotations

from typing import Optional

from flask import Flask, jsonify, request

from .config import AppConfig, load_config
from .inference import FIRGenerator
from .rule_based import extract_entities


def create_app(
    config: Optional[AppConfig] = None,
    generator: Optional[FIRGenerator] = None,
    prefer_model: bool = True,
) -> Flask:
    """Application factory. Injecting ``generator`` is handy for testing."""
    config = config or load_config()
    app = Flask(__name__)

    # The generator is created lazily on first use so app import stays cheap
    # and model loading failures don't crash startup.
    state: dict[str, Optional[FIRGenerator]] = {"generator": generator}

    def get_generator() -> FIRGenerator:
        if state["generator"] is None:
            state["generator"] = FIRGenerator(config=config, prefer_model=prefer_model)
        return state["generator"]

    @app.get("/health")
    def health():
        gen = state["generator"]
        return jsonify(
            {
                "status": "ok",
                "model_loaded": bool(gen and gen.using_model),
                "base_model": config.model.base_model,
            }
        )

    @app.post("/generate")
    def generate():
        payload = request.get_json(silent=True) or {}
        raw = payload.get("raw_statement", "")
        if not isinstance(raw, str) or not raw.strip():
            return jsonify({"error": "raw_statement (non-empty string) is required"}), 400
        gen = get_generator()
        complaint = gen.generate(raw)
        return jsonify(
            {
                "raw_statement": raw,
                "using_model": gen.using_model,
                "complaint": complaint.model_dump(),
                "document": complaint.to_document(),
            }
        )

    @app.post("/extract")
    def extract():
        payload = request.get_json(silent=True) or {}
        raw = payload.get("raw_statement", "")
        if not isinstance(raw, str) or not raw.strip():
            return jsonify({"error": "raw_statement (non-empty string) is required"}), 400
        return jsonify(
            {"raw_statement": raw, "extraction": extract_entities(raw).model_dump()}
        )

    return app
