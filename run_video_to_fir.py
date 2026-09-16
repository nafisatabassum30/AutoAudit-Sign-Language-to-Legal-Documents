import argparse
import re
import subprocess
import sys
import os


DEFAULT_INSTRUCTION = "ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন"


def run_command(command):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Command failed")

    return result.stdout
    


def extract_predicted_sentence(output_text):
    # Expected line:
    # 1. আপনার নাম কি | confidence=0.9999
    match = re.search(r"^\s*1\.\s*(.*?)\s*\|\s*confidence=", output_text, re.MULTILINE)

    if match:
        return match.group(1).strip()

    # Fallback pattern
    match = re.search(r"Predicted sentence:\s*(.*)", output_text)

    if match:
        return match.group(1).strip()

    raise ValueError("Could not extract predicted sentence from video model output.")


def extract_legal_complaint(output_text):
    marker = "=== Generated Legal Complaint ==="

    if marker in output_text:
        return output_text.split(marker, 1)[1].strip()

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: sign-language video to Bangla legal complaint"
    )

    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument(
        "--sign_model_dir",
        default="output/sign_sentence_full_gpu",
        help="Trained sign sentence model folder",
    )
    parser.add_argument(
        "--llm_adapter_dir",
        default="output/llm_lora_banglat5_final",
        help="Final BanglaT5 LoRA adapter folder",
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    args = parser.parse_args()

    print("\n==============================")
    print("STEP 1: Predicting Bangla sentence from video")
    print("==============================\n")

    video_command = [
        sys.executable,
        "vision/predict_from_video.py",
        "--video",
        args.video,
        "--model_dir",
        args.sign_model_dir,
        "--top_k",
        str(args.top_k),
    ]

    video_output = run_command(video_command)
    print(video_output)

    predicted_sentence = extract_predicted_sentence(video_output)

    print("\n==============================")
    print("Predicted Bangla Sentence")
    print("==============================")
    print(predicted_sentence)

    print("\n==============================")
    print("STEP 2: Generating Bangla FIR / Legal Complaint")
    print("==============================\n")

    llm_command = [
        sys.executable,
        "llm/infer_lora_t5.py",
        "--adapter_dir",
        args.llm_adapter_dir,
        "--instruction",
        args.instruction,
        "--transcript",
        predicted_sentence,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--num_beams",
        str(args.num_beams),
    ]

    llm_output = run_command(llm_command)
    legal_complaint = extract_legal_complaint(llm_output)

    print("\n==============================")
    print("Final Generated Legal Complaint")
    print("==============================\n")
    print(legal_complaint)


if __name__ == "__main__":
    main()