#!/usr/bin/env python3
"""Inference wrapper: informal Bangla text -> validated FIRRecord + document.

Usage
-----
    python inference.py --adapter outputs/fir-llm-lora \\
        --text "আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়"

The heavy ML dependencies (torch/transformers/peft) are imported lazily
inside :class:`FIRGenerator` so that the rest of this package (schema,
postprocessing, Flask API) can be imported/tested without a GPU or those
packages installed.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from src.fir_template import render_fir_document
from src.postprocess import FIRParseError, parse_fir_output
from src.prompts import build_chat_messages
from src.schema import FIRRecord


class FIRGenerator:
    """Loads a base model + optional LoRA adapter and generates FIR JSON."""

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"device_map": device_map, "trust_remote_code": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model

    def generate_raw(self, input_text: str) -> str:
        import torch

        messages = build_chat_messages(input_text)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate(self, input_text: str) -> FIRRecord:
        raw = self.generate_raw(input_text)
        result = parse_fir_output(raw, original_input_text=input_text)
        return result.record


def generate_fir_from_text(generate_fn, input_text: str) -> dict:
    """Shared helper used by both the CLI and the Flask API.

    ``generate_fn`` is any callable ``str -> str`` that returns raw model
    text (kept generic so tests can inject a stub instead of a real model).
    """
    raw = generate_fn(input_text)
    try:
        result = parse_fir_output(raw, original_input_text=input_text)
    except FIRParseError as e:
        return {"ok": False, "error": str(e), "raw_output": raw}

    document = render_fir_document(result.record)
    return {
        "ok": True,
        "fir_record": json.loads(result.record.model_dump_json()),
        "document_text": document,
        "repaired_json": result.repaired,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="hishab/titulm-llama-3.2-3b-v2.0")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter dir")
    parser.add_argument("--text", required=True, help="Informal Bangla input text")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    generator = FIRGenerator(
        base_model=args.base_model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
    )
    result = generate_fir_from_text(generator.generate_raw, args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
