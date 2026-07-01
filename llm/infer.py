# -*- coding: utf-8 -*-
"""Run the fine-tuned FIR-generation LLM: raw Bangla text (from the upstream
BdSL sign-recognition stage) in, a validated JSON complaint + a formatted,
FIR-ready legal document out.

Usage:
    python infer.py --adapter outputs/fir-qlora-adapter/final \\
        --text "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫"

    # merge LoRA into the base weights once, for faster/simpler serving:
    python infer.py --adapter outputs/fir-qlora-adapter/final --merge-and-save outputs/fir-merged
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoaudit_llm.dataset import format_messages
from autoaudit_llm.postprocess import render_fir_document, try_parse_fir_json
from autoaudit_llm.prompts import build_chat_messages


def load_pipeline(adapter_path: str, base_model: str = None, load_in_4bit: bool = False):
    """Load base model + LoRA adapter for generation.

    ``base_model`` is read from the adapter's config if not given explicitly.
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model is None:
        with open(os.path.join(adapter_path, "adapter_config.json"), "r", encoding="utf-8") as f:
            base_model = json.load(f)["base_model_name_or_path"]

    model_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_fir_json(model, tokenizer, raw_bangla_text: str, max_new_tokens: int = 512) -> str:
    messages = build_chat_messages(raw_bangla_text)
    prompt_text = format_messages(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Path to a saved LoRA adapter directory.")
    parser.add_argument("--base-model", default=None, help="Override base model (else read from adapter config).")
    parser.add_argument("--text", required=True, help="Raw Bangla text from the sign-recognition stage.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--merge-and-save", default=None, help="If set, merge LoRA into base weights and save here.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit (needs a GPU + bitsandbytes).")
    args = parser.parse_args()

    model, tokenizer = load_pipeline(args.adapter, args.base_model, args.load_in_4bit)

    if args.merge_and_save:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge_and_save)
        tokenizer.save_pretrained(args.merge_and_save)
        print(f"Merged model saved to {args.merge_and_save}")

    raw_output = generate_fir_json(model, tokenizer, args.text, args.max_new_tokens)
    print("--- raw model output ---")
    print(raw_output)

    complaint, error = try_parse_fir_json(raw_output)
    if complaint is None:
        print(f"\n[!] Could not parse a valid FIRComplaint from model output: {error}")
        return

    print("\n--- validated structured FIR ---")
    print(json.dumps(complaint.model_dump(mode="json"), ensure_ascii=False, indent=2))

    print("\n--- rendered FIR document ---")
    print(render_fir_document(complaint))


if __name__ == "__main__":
    main()
