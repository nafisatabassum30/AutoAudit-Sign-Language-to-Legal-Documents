"""Prompt templates for Bangla legal complaint generation."""

SYSTEM_PROMPT = """আপনি একজন বাংলাদেশি আইনি সহকারী।
আপনার কাজ হলো সাইন-টু-টেক্সট ইনপুট থেকে FIR-Ready অভিযোগ তৈরি করা।
আপনাকে অবশ্যই:
1) তথ্য বানিয়ে বলা যাবে না
2) অনুপস্থিত তথ্যে 'অজানা' লিখতে হবে
3) প্রথমে JSON ফরম্যাটে স্ট্রাকচার্ড ফিল্ড দিতে হবে
4) তারপর 'full_complaint_bn' এ পূর্ণাঙ্গ অভিযোগ লিখতে হবে
5) ভাষা হবে শালীন, আনুষ্ঠানিক, এবং আইনগতভাবে গ্রহণযোগ্য
"""

USER_TEMPLATE = """সাইন-ভিত্তিক বাংলা ইনপুট:
{sign_text_bn}

প্রাসঙ্গিক মেটাডাটা (ঐচ্ছিক):
{metadata}

নিচের নির্দিষ্ট JSON ফরম্যাটে উত্তর দিন:
{{
  "incident_date": "YYYY-MM-DD বা অজানা",
  "incident_time": "HH:MM বা অজানা",
  "location": "স্থান বা অজানা",
  "offense_type": "অপরাধের ধরন",
  "complainant_name": "অভিযোগকারীর নাম বা অজানা",
  "accused_name": "অভিযুক্তের নাম/বিবরণ বা অজানা",
  "summary_bn": "সংক্ষিপ্তসার",
  "full_complaint_bn": "পূর্ণাঙ্গ আনুষ্ঠানিক FIR-স্টাইল অভিযোগ",
  "requested_action_bn": "আইনানুগ পদক্ষেপের অনুরোধ"
}}
"""


def build_user_prompt(sign_text_bn: str, metadata: str = "অজানা") -> str:
    return USER_TEMPLATE.format(sign_text_bn=sign_text_bn.strip(), metadata=metadata)
