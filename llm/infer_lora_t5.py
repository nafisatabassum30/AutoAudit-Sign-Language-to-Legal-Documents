import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Bangla legal complaint using LoRA fine-tuned mT5/T5 model"
    )

    parser.add_argument(
        "--adapter_dir",
        type=Path,
        required=True,
        help="Path to LoRA adapter folder, e.g. output/llm_lora_test",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name/path. If empty, script reads from lora_training_info.json or adapter_config.json",
    )

    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Bangla sign-recognized sentence or incident text",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="Convert the following incident summary into a formal First Information Report (FIR) in Bengali.",
        help="Instruction used for the model prompt",
    )

    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--output_file", type=Path, default=None)

    return parser.parse_args()


def detect_base_model(adapter_dir: Path, user_base_model: str | None):
    if user_base_model:
        return user_base_model

    info_path = adapter_dir / "lora_training_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        base_model = info.get("base_model")
        if base_model:
            return base_model

    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model_name_or_path")
        if base_model:
            return base_model

    return "google/mt5-small"


def build_prompt(instruction: str, transcript: str):
    return (
        f"নির্দেশ: {instruction.strip()}\n"
        f"ইনপুট: {transcript.strip()}\n"
        "উত্তর:"
    )


def main():
    args = parse_args()

    adapter_dir = args.adapter_dir

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter folder not found: {adapter_dir}")

    base_model = detect_base_model(adapter_dir, args.base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Using device:", device)
    print("Base model:", base_model)
    print("LoRA adapter:", adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        dtype=dtype,
    )

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device)
    model.eval()

    prompt = build_prompt(args.instruction, args.transcript)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    print("\n=== Generated Legal Complaint ===\n")
    print(generated_text)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(generated_text, encoding="utf-8")
        print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()