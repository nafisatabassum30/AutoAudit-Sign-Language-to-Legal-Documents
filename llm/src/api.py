"""Flask API exposing the Bangla-FIR LLM as an HTTP service, so the
upstream ST-GNN sign-recognition stage (or any other client) can POST
raw Bangla text and receive a structured, FIR-ready legal complaint
back. This is the "Build Flask API to connect stages" step of the
AutoAudit development pipeline.

Endpoints:
    GET  /health
        -> {"status": "ok", "model_loaded": true}

    POST /generate-fir
        body: {"text": "আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকাল ৫টায়"}
        -> {
             "input_text": "...",
             "fir": {...FIR_FIELDS...},
             "document": "<formatted Bangla FIR text>"
           }

Configuration is via environment variables so the same image/service
can be pointed at different checkpoints without code changes:
    FIR_BASE_MODEL   - HF model id or local path of the base model.
    FIR_ADAPTER_PATH - Path to the fine-tuned LoRA adapter directory.
    FIR_MAX_NEW_TOKENS (optional, default 512)

Run:
    FIR_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    FIR_ADAPTER_PATH=outputs/bangla-fir-lora \\
    python -m src.api
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

try:
    from .fir_schema import FIRParseError
    from .infer import FIRGenerator
except ImportError:  # pragma: no cover
    from fir_schema import FIRParseError
    from infer import FIRGenerator

app = Flask(__name__)
_generator: FIRGenerator | None = None


def get_generator() -> FIRGenerator:
    global _generator
    if _generator is None:
        base_model = os.environ.get("FIR_BASE_MODEL")
        adapter_path = os.environ.get("FIR_ADAPTER_PATH")
        max_new_tokens = int(os.environ.get("FIR_MAX_NEW_TOKENS", "512"))
        if not base_model and not adapter_path:
            raise RuntimeError(
                "Set FIR_BASE_MODEL and/or FIR_ADAPTER_PATH environment variables "
                "before calling the API."
            )
        _generator = FIRGenerator(
            base_model=base_model, adapter_path=adapter_path, max_new_tokens=max_new_tokens
        )
    return _generator


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _generator is not None})


@app.post("/generate-fir")
def generate_fir():
    payload = request.get_json(silent=True) or {}
    input_text = (payload.get("text") or "").strip()
    if not input_text:
        return jsonify({"error": "Request body must include a non-empty 'text' field."}), 400

    try:
        generator = get_generator()
        record, document = generator.generate(input_text)
    except FIRParseError as exc:
        return jsonify({"error": f"Model output could not be parsed into a FIR: {exc}"}), 502
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"input_text": input_text, "fir": record.to_dict(), "document": document})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
