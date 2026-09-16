import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from prompt_templates import build_legal_complaint_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Bangla sign transcript into a legal complaint using a fine-tuned T5 model"
    )

    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to the fine-tuned T5 model directory",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--transcript",
        type=str,
        help="Bangla text transcript extracted from sign recognition",
    )
    group.add_argument(
        "--transcript_file",
        type=Path,
        help="Path to a text file containing the Bangla transcript",
    )

    parser.add_argument(
        "--additional_context",
        type=str,
        default=None,
        help="Optional extra context such as victim, location, or time details",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum new tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Optional path to save the generated complaint",
    )

    return parser.parse_args()


def read_transcript(args):
    if args.transcript_file:
        return args.transcript_file.read_text(encoding="utf-8").strip()

    return args.transcript.strip()


def main():
    args = parse_args()

    transcript = read_transcript(args)

    prompt = build_legal_complaint_prompt(
        transcript,
        additional_context=args.additional_context,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Loading fine-tuned T5 model from: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    model.to(device)
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            num_beams=1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    complaint_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    print("\n=== Generated Legal Complaint ===\n")
    print(complaint_text)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(complaint_text, encoding="utf-8")
        print(f"\nSaved generated complaint to: {args.output_file}")


if __name__ == "__main__":
    main()