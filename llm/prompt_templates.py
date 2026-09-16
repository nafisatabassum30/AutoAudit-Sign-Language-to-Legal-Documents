from typing import Optional

DEFAULT_INSTRUCTION = (
    "ভিডিওতে দেওয়া বর্ণনার ভিত্তিতে একটি সম্পূর্ণ বাংলাদেশী আইনি অভিযোগ লিখুন, যা FIR বা থানায় দায়েরের উপযোগী।"
)


def build_legal_complaint_prompt(
    transcript: str,
    additional_context: Optional[str] = None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> str:
    transcript = transcript.strip()
    prompt = f"নির্দেশ: {instruction}\nইনপুট: {transcript}"
    if additional_context:
        prompt += f"\nঅতিরিক্ত তথ্য: {additional_context.strip()}"
    prompt += "\nউত্তর:"
    return prompt


def format_training_example(
    transcript: str,
    complaint: str,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict:
    return {
        "instruction": instruction,
        "input": transcript.strip(),
        "output": complaint.strip(),
    }
