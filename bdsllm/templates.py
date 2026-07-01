from __future__ import annotations

from bdsllm.schema import IncidentFacts


def _value(value: str, fallback: str = "প্রদান করা হয়নি") -> str:
    value = value.strip()
    return value if value else fallback


def _sentence(value: str) -> str:
    value = value.strip()
    if not value:
        return "আইনগত ব্যবস্থা গ্রহণের অনুরোধ করছি।"
    return value if value.endswith(("।", ".", "?", "!")) else f"{value}।"


def render_legal_complaint(facts: IncidentFacts) -> str:
    """Render a FIR-style Bangla complaint without model generation.

    This is used to create weak labels for early experiments and as a safe
    fallback when an adapter/model is not available.
    """

    offense_type = _value(facts.offense_type, "অভিযোগ")
    accused = _value(facts.accused_name, "অজ্ঞাত ব্যক্তি")
    recognized_text = _value(facts.recognized_text, "প্রদান করা হয়নি")

    return f"""বরাবর
ভারপ্রাপ্ত কর্মকর্তা
সংশ্লিষ্ট থানা, বাংলাদেশ।

বিষয়: {offense_type} সংক্রান্ত অভিযোগ।

জনাব,
আমি {_value(facts.complainant_name, "অভিযোগকারী")}, ঠিকানা: {_value(facts.complainant_address)}। আমার ফোন নম্বর: {_value(facts.complainant_phone)}।

ঘটনার বিবরণ
ঘটনার তারিখ: {_value(facts.incident_date)}
ঘটনার সময়: {_value(facts.incident_time)}
ঘটনার স্থান: {_value(facts.incident_location)}
ঘটনার সংক্ষিপ্ত বিবরণ: {recognized_text}
অভিযুক্ত হিসেবে {accused} সম্পর্কে তথ্য পাওয়া গেছে। অভিযুক্তের অতিরিক্ত বিবরণ: {_value(facts.accused_details)}।

সংযুক্ত তথ্য
অপরাধের ধরন: {offense_type}
অতিরিক্ত প্রেক্ষাপট: {_value(facts.additional_context, "নেই")}

প্রার্থনা
উপরোক্ত ঘটনার বিষয়ে {_sentence(facts.requested_action)} প্রয়োজনীয় তদন্ত করার জন্য বিনীত অনুরোধ করছি।

অভিযোগকারীর তথ্য
নাম: {_value(facts.complainant_name)}
ঠিকানা: {_value(facts.complainant_address)}
ফোন: {_value(facts.complainant_phone)}
স্বাক্ষর: ____________________
তারিখ: {_value(facts.incident_date)}
""".strip()
