import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from vision.predict_from_video_v2 import extract_keypoints_from_video, load_model, predict


def detect_case_type(text: str) -> str:
    t = text.lower()
    if "মোবাইল" in text and ("নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "চুরি" in text):
        return "মোবাইল চুরি / ছিনতাই"
    if "ব্যাগ" in text and ("নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "চুরি" in text):
        return "ব্যাগ চুরি / ছিনতাই"
    if "হুমকি" in text or "ভয়" in text or "ভয়" in text:
        return "হুমকি প্রদান"
    if "আঘাত" in text or "ব্যথা" in text:
        return "শারীরিক আঘাত"
    if "টাকা" in text and ("নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "আত্মসাৎ" in text):
        return "টাকা নেওয়া / আর্থিক ক্ষতি"
    return "আইনগত অভিযোগ"


def build_fir(text: str) -> str:
    case_type = detect_case_type(text)
    return f"""ফৌজদারি অভিযোগ / FIR

মামলা নং: ____________________

ঘটনার তারিখ: ____________________
ঘটনার সময়: ____________________
ঘটনার স্থান: ____________________

বাদীর নাম: ____________________
বাদীর ঠিকানা: ____________________

ঘটনার ধরন:
{case_type}

ঘটনার বিবরণ:
দায়েরকারী অভিযোগ করেন যে {text}

অভিযুক্ত ব্যক্তি বা প্রতিষ্ঠান:
অজ্ঞাত / উল্লিখিত ব্যক্তি

প্রমাণের বর্ণনা:
প্রয়োজনীয় সাক্ষ্য, ভিডিও ফুটেজ, প্রত্যক্ষদর্শীর বক্তব্য বা অন্যান্য প্রমাণ তদন্তসাপেক্ষে সংযোজনীয়।

আবেদন:
উক্ত ঘটনার বিষয়ে দ্রুত আইনানুগ ব্যবস্থা গ্রহণের জন্য বিনীত অনুরোধ করা হলো।

স্বাক্ষর: ____________________
তারিখ: ____________________
"""


def generate_llm_fir(adapter_dir: Path, transcript: str):
    # Match project behavior used by llm/infer_lora_t5.py
    info_path = adapter_dir / "lora_training_info.json"
    base_model = None
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        base_model = info.get("base_model")
    if not base_model:
        cfg = adapter_dir / "adapter_config.json"
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
            base_model = data.get("base_model_name_or_path")
    base_model = base_model or "google/mt5-small"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(base_model), adapter_dir)
    model.to(device)
    model.eval()

    prompt = f"নির্দেশ: ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন\nইনপুট: {transcript.strip()}\nউত্তর:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Run video sign recognition and generate crime FIR.")
    parser.add_argument("--video", default="data/Test_data/single_videos/0.mp4")
    parser.add_argument("--model_dir", default="output/sign_sentence_v2_validated")
    parser.add_argument("--adapter_dir", default="output/llm_lora_banglat5_final")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--out_dir", default="outputs/video_crime_fir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, id_to_label = load_model(args.model_dir, device)
    keypoints = extract_keypoints_from_video(args.video, num_frames=32)
    results = predict(model, keypoints, id_to_label, device, top_k=args.top_k)
    top_sentence = results[0][0]
    crime_type = detect_case_type(top_sentence)

    print("Top predicted sentence:", top_sentence)
    print("Crime type:", crime_type)

    llm_fir = generate_llm_fir(Path(args.adapter_dir), top_sentence)
    final_fir = llm_fir if "ফৌজদারি অভিযোগ / FIR" in llm_fir else build_fir(top_sentence)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.video).stem}_fir.txt"
    out_file.write_text(final_fir, encoding="utf-8")

    print("\n=== Generated FIR ===")
    print(final_fir)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
