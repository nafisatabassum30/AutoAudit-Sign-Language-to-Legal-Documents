#!/usr/bin/env python3
"""Inference for the BdSL-raw-text -> FIR-JSON LLM stage.

Given raw Bangla text produced by the upstream ST-GNN sign-recognition stage,
this loads the base model (+ optional LoRA adapter), generates a structured
FIR JSON object, validates/repairs it, and renders the formatted Bangla FIR
document.

Example:
    python -m src.infer --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter checkpoints/fir-lora/final \\
        --text "আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fir_parser import parse_and_validate, render_document  # noqa: E402
from src.prompts import build_chat_messages, render_prompt_plain  # noqa: E402


class FIRGenerator:
    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(self.device)
        self.model.eval()

    def _build_prompt_ids(self, raw_signed_text: str) -> torch.Tensor:
        if getattr(self.tokenizer, "chat_template", None):
            messages = build_chat_messages(raw_signed_text)
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            prompt_text = render_prompt_plain(raw_signed_text)
            ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        # Depending on the transformers version, apply_chat_template(..., return_tensors="pt")
        # may return a raw tensor or a BatchEncoding wrapping one.
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return ids.to(self.device)

    @torch.inference_mode()
    def generate_raw(self, raw_signed_text: str) -> str:
        input_ids = self._build_prompt_ids(raw_signed_text)
        attention_mask = torch.ones_like(input_ids)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        completion_ids = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)

    def generate_fir(self, raw_signed_text: str) -> dict:
        raw_output = self.generate_raw(raw_signed_text)
        fir = parse_and_validate(raw_output)
        return {"fir": fir, "document": render_document(fir), "llm_raw_output": raw_output}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None, help="Path to a fine-tuned LoRA adapter directory")
    parser.add_argument("--text", required=True, help="Raw Bangla text from the sign-recognition stage")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--json-only", action="store_true", help="Print only the FIR JSON, not the document")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = FIRGenerator(args.base_model, args.adapter, max_new_tokens=args.max_new_tokens)
    result = generator.generate_fir(args.text)
    if args.json_only:
        print(json.dumps(result["fir"], ensure_ascii=False, indent=2))
    else:
        print(result["document"])


if __name__ == "__main__":
    main()
