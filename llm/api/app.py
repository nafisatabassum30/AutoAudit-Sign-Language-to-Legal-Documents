"""
Flask REST API — BdSL Bangla FIR Generator

Endpoints:
  POST /generate          — Generate FIR from informal Bangla text
  POST /generate/batch    — Batch generation
  POST /generate/stream   — Server-Sent Events streaming
  GET  /health            — Health check
  GET  /model/info        — Model metadata
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import FIRGenerator, InferenceConfig
from src.utils import setup_logging, get_gpu_info, normalize_bangla

setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ---------------------------------------------------------------------------
# Generator (lazy-loaded on first request)
# ---------------------------------------------------------------------------

_generator: FIRGenerator | None = None


def get_generator() -> FIRGenerator:
    global _generator
    if _generator is None:
        cfg = InferenceConfig(
            model_path=os.getenv("MODEL_PATH", "models/checkpoints/final_adapter"),
            base_model_name=os.getenv("BASE_MODEL", "unsloth/llama-3-8b-bnb-4bit"),
            load_in_4bit=os.getenv("LOAD_IN_4BIT", "true").lower() == "true",
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
        )
        _generator = FIRGenerator(cfg)
        _generator.load()
        logger.info("Generator initialised.")
    return _generator


# ---------------------------------------------------------------------------
# Request validation helper
# ---------------------------------------------------------------------------

def _require_json_field(field: str):
    data = request.get_json(silent=True)
    if not data:
        return None, (jsonify({"error": "JSON body required"}), 400)
    value = data.get(field)
    if not value or not isinstance(value, str) or not value.strip():
        return None, (jsonify({"error": f"'{field}' field is required and must be a non-empty string"}), 400)
    return value.strip(), None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    gpu = get_gpu_info()
    return jsonify(
        {
            "status": "ok",
            "model_loaded": _generator is not None and _generator._loaded,
            "gpu": gpu,
        }
    )


@app.get("/model/info")
def model_info():
    gen = get_generator()
    return jsonify(
        {
            "model_path": gen.config.model_path,
            "base_model": gen.config.base_model_name,
            "load_in_4bit": gen.config.load_in_4bit,
            "max_new_tokens": gen.config.max_new_tokens,
            "temperature": gen.config.temperature,
        }
    )


@app.post("/generate")
def generate():
    """
    Request body:
        { "text": "<informal Bangla text>" }

    Response:
        {
          "input": "...",
          "fir_text": "...",
          "entities": { ... },
          "latency_ms": 1234
        }
    """
    text, err = _require_json_field("text")
    if err:
        return err

    text = normalize_bangla(text)
    start = time.perf_counter()

    try:
        gen = get_generator()
        result = gen.generate_structured(text)
    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": str(e)}), 500

    elapsed_ms = round((time.perf_counter() - start) * 1000)
    result["latency_ms"] = elapsed_ms
    return jsonify(result)


@app.post("/generate/batch")
def generate_batch():
    """
    Request body:
        { "texts": ["text1", "text2", ...] }

    Response:
        { "results": [ { "input": ..., "fir_text": ..., "entities": ... }, ... ] }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    texts = data.get("texts")
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    if len(texts) > 20:
        return jsonify({"error": "Batch size limit is 20"}), 400

    try:
        gen = get_generator()
        results = [gen.generate_structured(normalize_bangla(t)) for t in texts]
    except Exception as e:
        logger.exception("Batch generation failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({"results": results})


@app.post("/generate/stream")
def generate_stream():
    """
    Server-Sent Events endpoint.
    Request body: { "text": "<informal Bangla text>" }
    Streams tokens as SSE events; final event is `data: [DONE]`.
    """
    text, err = _require_json_field("text")
    if err:
        return err

    text = normalize_bangla(text)
    gen = get_generator()

    @stream_with_context
    def event_stream():
        try:
            for token in gen.stream(text):
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as e:
            logger.exception("Streaming failed")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting FIR API server on port %d (debug=%s)", port, debug)
    # Pre-load model on startup
    get_generator()
    app.run(host="0.0.0.0", port=port, debug=debug)
