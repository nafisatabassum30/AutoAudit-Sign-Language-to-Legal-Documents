import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Bangla FIR complaints using a fine-tuned seq2seq model")
    parser.add_argument("--model_dir", type=Path, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Bangla instruction or incident description")
    parser.add_argument("--max_length", type=int, default=260, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    return parser.parse_args()


def build_prompt(prompt_text: str) -> str:
    return (
        "নির্দেশ: আপনি একটি পূর্ণাঙ্গ বাংলা FIR/অভিযোগপত্র তৈরি করুন। শুধু FIR/অভিযোগপত্রের বিষয়বস্তু লিখুন, কোনো অতিরিক্ত ব্যাখ্যা দেবেন না.\n"
        f"ঘটনার বিবরণ: {prompt_text}\n"
        "FIR/অভিযোগপত্র:"
    )


def build_fir_document(prompt_text: str) -> str:
    cleaned_prompt = re.sub(r"\s+", " ", str(prompt_text or "").strip())
    return f"""ফৌজদারি অভিযোগ / FIR

মামলা নং: __________
ঘটনার তারিখ: __________
ঘটনার সময়: __________
ঘটনার স্থান: __________

বাদীর নাম: __________
বাদীর ঠিকানা: __________

ঘটনার বিবরণ:
দায়েরকারী অভিযোগ করেন যে {cleaned_prompt}।

অভিযুক্ত ব্যক্তি বা প্রতিষ্ঠান: __________
প্রমাণের বর্ণনা: __________

বাদী এই ঘটনার বিষয়ে দ্রুত আইনানুগ ব্যবস্থা গ্রহণের জন্য বিনীত অনুরোধ করেন।

স্বাক্ষর: __________
তারিখ: __________"""


def clean_generated_text(text: str) -> str:
    text = text.strip()
    if "FIR/অভিযোগপত্র:" in text:
        text = text.split("FIR/অভিযোগপত্র:", 1)[1].strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    args = parse_args()
    try:
        print(f"Loading fine-tuned model from {args.model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, torch_dtype=torch.float32, trust_remote_code=True)
        model.eval()

        prompt = build_prompt(args.prompt)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            num_beams=4,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cleaned = clean_generated_text(result)
        if cleaned and len(cleaned) > 40 and "অভিযোগ" in cleaned:
            print("\n=== Generated Complaint ===\n")
            print(cleaned)
            return
    except Exception as exc:
        print(f"Model generation failed: {exc}")

    print("\n=== Generated Complaint ===\n")
    print(build_fir_document(args.prompt))


if __name__ == "__main__":
    main()
