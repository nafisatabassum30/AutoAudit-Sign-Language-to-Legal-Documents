"""Inference for Bangla FIR-style complaint generation."""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import SYSTEM_PROMPT, build_user_prompt
from schema import LegalComplaint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Bangla legal complaint from sign-text input.")
    parser.add_argument("--model_path", type=str, required=True, help="Fine-tuned model or adapter path.")
    parser.add_argument("--sign_text_bn", type=str, required=True, help="Bangla text from sign recognition module.")
    parser.add_argument("--metadata", type=str, default="অজানা", help="Optional metadata as plain text/JSON string.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output does not contain JSON.\nRaw output:\n{text}")
    return match.group(0)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    user_prompt = build_user_prompt(args.sign_text_bn, args.metadata)
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assistant_part = decoded.split("assistant", maxsplit=1)[-1]
    json_text = extract_json_block(assistant_part)
    parsed = json.loads(json_text)
    validated = LegalComplaint.model_validate(parsed)

    print(json.dumps(validated.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
