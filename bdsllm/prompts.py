from __future__ import annotations

from bdsllm.schema import IncidentFacts


SYSTEM_PROMPT = """আপনি বাংলাদেশের আইনগত অভিযোগ লেখার সহায়ক একজন বাংলা ভাষার সহকারী।
ইনপুটে BdSL সাইন-রিকগনিশন থেকে পাওয়া অপরিশোধিত বাংলা টেক্সট এবং ঘটনার তথ্য থাকবে।
আপনার কাজ হলো তথ্য পরিবর্তন না করে পরিষ্কার, সম্মানজনক, FIR-উপযোগী অভিযোগ খসড়া লেখা।
অনুমান করে নতুন অপরাধ, নাম, তারিখ, সময় বা স্থান যোগ করবেন না। তথ্য অনুপস্থিত হলে "প্রদান করা হয়নি" লিখুন।
"""


INSTRUCTION_TEMPLATE = """নিচের তথ্য ব্যবহার করে বাংলায় একটি আইনগত অভিযোগ খসড়া তৈরি করুন।

সাইন থেকে পাওয়া বাংলা টেক্সট:
{recognized_text}

অভিযোগকারীর নাম: {complainant_name}
অভিযোগকারীর ঠিকানা: {complainant_address}
অভিযোগকারীর ফোন: {complainant_phone}
ঘটনার তারিখ: {incident_date}
ঘটনার সময়: {incident_time}
ঘটনার স্থান: {incident_location}
অভিযুক্তের নাম: {accused_name}
অভিযুক্তের বিবরণ: {accused_details}
অপরাধের ধরন: {offense_type}
অতিরিক্ত প্রেক্ষাপট: {additional_context}
চাওয়া ব্যবস্থা: {requested_action}

আউটপুটে এই শিরোনামগুলো রাখুন:
১. বরাবর
২. বিষয়
৩. জনাব
৪. ঘটনার বিবরণ
৫. সংযুক্ত তথ্য
৬. প্রার্থনা
৭. অভিযোগকারীর তথ্য
"""


def build_instruction(facts: IncidentFacts) -> str:
    return f"{SYSTEM_PROMPT.strip()}\n\n{INSTRUCTION_TEMPLATE.format(**facts.to_prompt_fields()).strip()}"


def build_chat_messages(facts: IncidentFacts) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(**facts.to_prompt_fields()).strip()},
    ]
