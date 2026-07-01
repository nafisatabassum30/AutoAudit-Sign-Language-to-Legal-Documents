"""Command line entrypoints for the AutoAudit Bangla FIR LLM.

Subcommands
-----------
generate-data  Generate + split a synthetic FIR dataset into train/eval JSONL.
train          Run QLoRA fine-tuning (requires the training extras + GPU).
infer          Convert a single raw Bangla statement into an FIR complaint.
serve          Launch the Flask API.

Usage::

    python -m autoaudit_llm.cli generate-data --config configs/default.yaml
    python -m autoaudit_llm.cli infer --text "আমার মানিব্যাগ চুরি উত্তরা বিকেল ৫টা"
    python -m autoaudit_llm.cli serve
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .config import load_config
from .data_generation import generate_dataset
from .dataset import split_dataset, write_jsonl


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


# --- generate-data -----------------------------------------------------------
def generate_data_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Bangla FIR dataset")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--num", type=int, default=None, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    _setup_logging()
    config = load_config(args.config)
    num = args.num or config.data.num_synthetic_examples
    records = generate_dataset(num, seed=args.seed)
    train, eval_ = split_dataset(records, config.data.eval_split, seed=args.seed)
    write_jsonl(train, config.data.train_file)
    write_jsonl(eval_, config.data.eval_file)
    logging.info(
        "Wrote %d train -> %s and %d eval -> %s",
        len(train), config.data.train_file, len(eval_), config.data.eval_file,
    )
    return 0


# --- train -------------------------------------------------------------------
def train_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the Bangla FIR LLM")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args(argv)

    _setup_logging()
    config = load_config(args.config)
    try:
        from .train import train as run_train
    except Exception as exc:  # pragma: no cover
        logging.error("Training deps missing: %s", exc)
        logging.error("Install with: pip install -r requirements-train.txt")
        return 1
    adapter_dir = run_train(config)
    logging.info("Training complete. Adapter saved to %s", adapter_dir)
    return 0


# --- infer -------------------------------------------------------------------
def infer_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an FIR from a Bangla statement")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--text", required=True, help="Raw Bangla statement")
    parser.add_argument(
        "--no-model", action="store_true", help="Force rule-based baseline"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of doc")
    args = parser.parse_args(argv)

    _setup_logging()
    from .inference import FIRGenerator

    config = load_config(args.config)
    gen = FIRGenerator(config=config, prefer_model=not args.no_model)
    complaint = gen.generate(args.text)
    if args.json:
        print(complaint.model_dump_json(indent=2))
    else:
        print(complaint.to_document())
    return 0


# --- serve -------------------------------------------------------------------
def serve_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Bangla FIR LLM API")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--no-model", action="store_true", help="Force rule-based baseline"
    )
    args = parser.parse_args(argv)

    _setup_logging()
    from .api import create_app

    config = load_config(args.config)
    app = create_app(config=config, prefer_model=not args.no_model)
    host = args.host or config.api.host
    port = args.port or config.api.port
    app.run(host=host, port=port)
    return 0


# --- dispatcher --------------------------------------------------------------
_COMMANDS = {
    "generate-data": generate_data_main,
    "train": train_main,
    "infer": infer_main,
    "serve": serve_main,
}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _COMMANDS:
        print("Usage: python -m autoaudit_llm.cli <command> [options]")
        print("Commands: " + ", ".join(_COMMANDS))
        return 1
    command, rest = argv[0], argv[1:]
    return _COMMANDS[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
