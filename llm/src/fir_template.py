"""Render a validated :class:`FIRRecord` into a formal, FIR-ready Bangla
legal complaint document (plain text, submission-ready).

Keeping template rendering deterministic and separate from the LLM means
the *legal boilerplate/formatting* never depends on model generation
quality -- the LLM only needs to extract structured facts, and this module
guarantees a consistently formatted document every time.
"""
from __future__ import annotations

from datetime import date

from .schema import FIRRecord, Person


def _fmt_person(p: Person | None, label: str) -> str:
    if p is None:
        return f"{label}: তথ্য উল্লেখ নেই"
    lines = [f"{label}:"]
    lines.append(f"  নাম: {p.name or 'অজ্ঞাত'}")
    if p.address:
        lines.append(f"  ঠিকানা: {p.address}")
    if p.phone:
        lines.append(f"  মোবাইল: {p.phone}")
    if p.nid:
        lines.append(f"  জাতীয় পরিচয়পত্র: {p.nid}")
    if p.relation_to_victim:
        lines.append(f"  সম্পর্ক: {p.relation_to_victim}")
    return "\n".join(lines)


def render_fir_document(record: FIRRecord, *, filing_date: str | None = None) -> str:
    """Render ``record`` as a plain-text, FIR-ready Bangla complaint letter."""
    record = record.with_defaults_filled()
    filing_date = filing_date or date.today().isoformat()

    sections = record.penal_code_sections or ["প্রযোজ্য আইনের ধারা উল্লেখ করা হয়নি"]
    items_lines = []
    for item in record.stolen_or_damaged_items:
        qty = f"{item.quantity} x " if item.quantity and item.quantity != 1 else ""
        value = (
            f" (আনুমানিক মূল্য: {item.estimated_value_bdt:,.0f} টাকা)"
            if item.estimated_value_bdt
            else ""
        )
        items_lines.append(f"  - {qty}{item.description}{value}")
    items_block = "\n".join(items_lines) if items_lines else "  - উল্লেখ নেই"

    witnesses_block = (
        "\n".join(f"  - {w}" for w in record.witnesses)
        if record.witnesses
        else "  - উল্লেখ নেই"
    )

    accused_block = (
        "অভিযুক্ত: অজ্ঞাত/অচিহ্নিত ব্যক্তি"
        if record.accused_unknown or record.accused is None
        else _fmt_person(record.accused, "অভিযুক্তের তথ্য")
    )

    doc = f"""
প্রথম তথ্য প্রতিবেদন (FIR) - অভিযোগপত্র
=========================================

তারিখ (দাখিলের): {filing_date}
থানা: {record.police_station or 'উল্লেখ নেই'}
জেলা: {record.district or 'উল্লেখ নেই'}

বিষয়: {record.offense_type.value} সংক্রান্ত অভিযোগ

{_fmt_person(record.complainant, "অভিযোগকারীর তথ্য")}

{_fmt_person(record.victim, "ভুক্তভোগীর তথ্য")}

{accused_block}

ঘটনার বিবরণ:
  ঘটনার তারিখ: {record.incident_date or 'উল্লেখ নেই'}
  ঘটনার সময়: {record.incident_time or 'উল্লেখ নেই'}
  ঘটনাস্থল: {record.incident_location}

সংশ্লিষ্ট সম্পত্তি/জিনিসপত্র:
{items_block}

সাক্ষী:
{witnesses_block}

সম্ভাব্য প্রযোজ্য আইনের ধারা (আইনজীবী/তদন্তকারী কর্তৃক নিশ্চিত করতে হবে):
{chr(10).join('  - ' + s for s in sections)}

অভিযোগের বিস্তারিত বর্ণনা:
{record.narrative_bn}

--------------------------------------------------
অতএব, উপরোক্ত বিষয়ে যথাযথ ব্যবস্থা গ্রহণের জন্য বিনীত আবেদন জানানো হলো।

অভিযোগকারীর স্বাক্ষর/টিপসই: ____________________
""".strip()
    return doc
