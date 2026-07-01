import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Bangla legal complaints using a fine-tuned LLM")
    parser.add_argument("--model_dir", type=Path, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Bangla instruction or incident description")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    return parser.parse_args()


def build_prompt(prompt_text: str) -> str:
    return f"নির্দেশ: {prompt_text}\nউত্তর:"


def main():
    args = parse_args()
    print(f"Loading fine-tuned model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    prompt = build_prompt(args.prompt)
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
    print("\n=== Generated Complaint ===\n")
    print(result)


if __name__ == "__main__":
    main()
