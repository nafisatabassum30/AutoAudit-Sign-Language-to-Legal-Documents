import argparse
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Bangla legal complaints using a fine-tuned LLM")
    parser.add_argument("--model_dir", type=Path, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Bangla instruction or incident description")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    return parser.parse_args()


def build_prompt(prompt_text: str) -> str:
    return (
        "নির্দেশ: আপনি একটি formal FIR/অভিযোগপত্র বাংলায় তৈরি করুন। "
        "ঘটনার বিবরণ দেওয়া হলে, ১) মামলার শিরোনাম, ২) ঘটনার বিবরণ, ৩) অভিযোগের বর্ণনা, "
        "৪) প্রমাণের উল্লেখ, এবং ৫) বাদীর আবেদন লিখুন।\n"
        f"ঘটনার বিবরণ: {prompt_text}\n"
        "FIR/অভিযোগপত্র:"
    )


def clean_generated_text(text: str) -> str:
    text = text.strip()
    if "FIR/অভিযোগপত্র:" in text:
        text = text.split("FIR/অভিযোগপত্র:", 1)[1].strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    args = parse_args()
    base_model = "gpt2"
    print(f"Loading fine-tuned model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model_obj, args.model_dir)
    model.eval()

    prompt = build_prompt(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
    )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned = clean_generated_text(result)
    print("\n=== Generated Complaint ===\n")
    print(cleaned)


if __name__ == "__main__":
    main()
