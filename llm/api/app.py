#!/usr/bin/env python3
"""Flask API exposing the fine-tuned Bangla FIR-generation LLM.

This is the integration point between the ST-GNN sign-recognition stage
(which produces raw Bangla text) and the final FIR document. It is designed
to be called by the Flask API layer described in the project's overall
pipeline ("Build Flask API to connect stages").

Endpoints
---------
    GET  /health
    POST /api/v1/generate-fir   {"text": "<recognized bangla text>"}

Run standalone (loads the real model, requires GPU + ML deps):
    python api/app.py --base-model hishab/titulm-llama-3.2-3b-v2.0 \\
        --adapter outputs/fir-llm-lora
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import generate_fir_from_text  # noqa: E402

logger = logging.getLogger("fir_llm_api")


def create_app(generate_fn: Optional[Callable[[str], str]] = None) -> Flask:
    """App factory.

    ``generate_fn`` is a ``str -> str`` callable that returns *raw* model
    text for a given informal Bangla input. Passing it explicitly (rather
    than always constructing a real :class:`~inference.FIRGenerator`) keeps
    this module trivially unit-testable and lets deployments swap in
    whichever model-serving backend they prefer (local HF, vLLM, TGI, ...).
    """
    app = Flask(__name__)
    app.config["GENERATE_FN"] = generate_fn

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": app.config["GENERATE_FN"] is not None})

    @app.post("/api/v1/generate-fir")
    def generate_fir():
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "Field 'text' is required and must be non-empty."}), 400

        fn = app.config["GENERATE_FN"]
        if fn is None:
            return (
                jsonify({"ok": False, "error": "Model not loaded on this server instance."}),
                503,
            )

        try:
            result = generate_fir_from_text(fn, text)
        except Exception:  # noqa: BLE001 - surface as a clean 500 to API clients
            logger.exception("Unhandled error while generating FIR for input: %r", text)
            return jsonify({"ok": False, "error": "Internal error while generating FIR."}), 500

        status_code = 200 if result.get("ok") else 422
        return jsonify(result), status_code

    return app


def _build_real_generate_fn(base_model: str, adapter: Optional[str], no_4bit: bool):
    from inference import FIRGenerator

    generator = FIRGenerator(base_model=base_model, adapter_path=adapter, load_in_4bit=not no_4bit)
    return generator.generate_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=os.environ.get("FIR_LLM_BASE_MODEL", "hishab/titulm-llama-3.2-3b-v2.0"))
    parser.add_argument("--adapter", default=os.environ.get("FIR_LLM_ADAPTER_PATH"))
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_fn = _build_real_generate_fn(args.base_model, args.adapter, args.no_4bit)
    app = create_app(generate_fn=generate_fn)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
