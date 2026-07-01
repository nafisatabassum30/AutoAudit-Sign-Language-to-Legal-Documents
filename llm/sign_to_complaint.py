import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_templates import build_legal_complaint_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Bangla sign transcript into a legal complaint using the fine-tuned LLM")
    parser.add_argument("--model_dir", type=Path, required=True, help="Path to the fine-tuned model directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--transcript", type=str, help="Bangla text transcript extracted from sign recognition")
    group.add_argument("--transcript_file", type=Path, help="Path to a text file containing the Bangla transcript")
    parser.add_argument("--additional_context", type=str, default=None, help="Optional extra context such as victim, location, or time details")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--output_file", type=Path, default=None, help="Optional path to save the generated complaint")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.transcript_file:
        transcript = args.transcript_file.read_text(encoding="utf-8").strip()
    else:
        transcript = args.transcript.strip()

    prompt = build_legal_complaint_prompt(transcript, additional_context=args.additional_context)

    print(f"Loading fine-tuned model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
    )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    complaint_text = result[len(prompt) :].strip() if result.startswith(prompt) else result

    print("\n=== Generated Legal Complaint ===\n")
    print(complaint_text)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(complaint_text, encoding="utf-8")
        print(f"Saved generated complaint to {args.output_file}")


if __name__ == "__main__":
    main()
