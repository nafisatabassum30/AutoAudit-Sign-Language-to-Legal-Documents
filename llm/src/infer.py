"""Inference for the fine-tuned Bangla-FIR LLM.

Given raw Bangla text (produced upstream by the ST-GNN sign-recognition
stage), this module loads the base model + LoRA adapter and generates a
structured FIR JSON object, then renders it into a submission-ready
Bangla FIR document.

CLI usage:
    python -m src.infer --adapter outputs/bangla-fir-lora \\
        --text "আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকাল ৫টায়"

Library usage:
    from src.infer import FIRGenerator
    gen = FIRGenerator(base_model="Qwen/Qwen2.5-7B-Instruct", adapter_path="outputs/bangla-fir-lora")
    record, document = gen.generate("আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকাল ৫টায়")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .fir_schema import FIRParseError, FIRRecord, parse_fir_output, render_fir_document
    from .prompts import build_prompt
except ImportError:  # pragma: no cover
    from fir_schema import FIRParseError, FIRRecord, parse_fir_output, render_fir_document
    from prompts import build_prompt


class FIRGenerator:
    """Wraps a base model (+ optional LoRA adapter) for FIR generation."""

    def __init__(
        self,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
    ):
        if base_model is None and adapter_path is None:
            raise ValueError("Provide at least one of base_model or adapter_path")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        tokenizer_source = adapter_path if (adapter_path and self._has_tokenizer(adapter_path)) else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        model_source = base_model or adapter_path
        model = AutoModelForCausalLM.from_pretrained(model_source, trust_remote_code=trust_remote_code)

        if adapter_path is not None and base_model is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def _has_tokenizer(path: str) -> bool:
        p = Path(path)
        return p.exists() and (p / "tokenizer_config.json").exists()

    def _max_context_length(self) -> Optional[int]:
        config = getattr(self.model, "config", None)
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and 0 < model_max_length < 1_000_000:
            return model_max_length
        return None

    @torch.inference_mode()
    def generate_raw(self, input_text: str, **generate_kwargs) -> str:
        prompt = build_prompt(self.tokenizer, input_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_new_tokens = generate_kwargs.pop("max_new_tokens", self.max_new_tokens)
        context_limit = self._max_context_length()
        if context_limit is not None:
            prompt_len = inputs["input_ids"].shape[1]
            headroom = max(1, context_limit - prompt_len)
            if max_new_tokens > headroom:
                max_new_tokens = headroom

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_kwargs.update(generate_kwargs)

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate(self, input_text: str, **generate_kwargs) -> Tuple[FIRRecord, str]:
        """Generate a FIRRecord and its rendered document for the given
        raw input text. Raises FIRParseError if the model output could
        not be parsed."""

        raw_output = self.generate_raw(input_text, **generate_kwargs)
        try:
            record = parse_fir_output(raw_output)
        except FIRParseError:
            raise FIRParseError(f"Could not parse model output as FIR JSON: {raw_output!r}")
        return record, render_fir_document(record)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None, help="Path to a saved LoRA adapter directory.")
    parser.add_argument("--text", type=str, required=True, help="Raw Bangla input text.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--json-only", action="store_true", help="Print only the FIR JSON, not the rendered document.")
    args = parser.parse_args()

    generator = FIRGenerator(
        base_model=args.base_model, adapter_path=args.adapter, max_new_tokens=args.max_new_tokens
    )
    record, document = generator.generate(args.text)

    if args.json_only:
        print(record.to_json())
    else:
        print(record.to_json())
        print()
        print(document)


if __name__ == "__main__":
    _cli()
