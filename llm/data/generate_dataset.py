# -*- coding: utf-8 -*-
"""Synthesize a bootstrap instruction-tuning dataset for the FIR-generation LLM.

Produces JSONL rows of the form::

    {"raw_text": "<informal Bangla text mimicking ST-GNN sign output>",
     "target": {...FIRComplaint fields...}}

Run:
    python generate_dataset.py --n-per-template 12 --seed 42 \\
        --out llm/data/processed/synthetic_raw.jsonl

This is a *bootstrap* generator meant to unblock LLM training/evaluation
before the real 1,000+ collected FIR templates (BLAST / police) are folded
in. See ``llm/data/vocab.py`` module docstring for how to extend this with
real data later.

Design principle -- no hallucinated facts: every fact stated in ``target``
(a specific date, time, accused identity, amount, item, victim...) MUST also
be grounded in ``raw_text``. If a chosen raw phrasing doesn't happen to
mention a slot that the offense's narrative needs, the slot value is
appended to ``raw_text`` (see ``_ground_raw_text``). This mirrors the system
prompt's explicit instruction to the model: never invent facts that weren't
in the input.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import (  # noqa: E402
    ACCUSED_NAMES,
    AMOUNTS,
    COMPLAINANT_NAMES,
    HUSBAND_NAMES,
    ITEMS,
    LOCATIONS,
    THANA_BY_LOCATION,
    TIMES,
    VICTIM_NAMES,
    WITNESS_NAME_POOL,
    maybe,
    random_recent_date,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoaudit_llm.schema import OffenseType  # noqa: E402

# Basic incident metadata every FIR needs, regardless of whether the
# free-text narrative happens to repeat it -- always grounded in raw_text.
_ALWAYS_GROUNDED_SLOTS = ["date", "time", "location"]
# Offense-specific facts (a name, an amount...) that are only asserted in the
# target when they're semantically relevant to that offense category, so
# they're only grounded when the narrative actually uses them.
_CONDITIONALLY_GROUNDED_SLOTS = ["item", "amount", "accused", "victim"]


@dataclass
class OffenseTemplate:
    offense_type: OffenseType
    raw_templates: List[str]
    narrative_template: str
    uses_item: bool = False
    uses_amount: bool = False
    uses_victim: bool = False
    uses_accused: bool = False
    accused_pool: Optional[List[str]] = None
    property_field: Optional[Callable[[str], str]] = None


TEMPLATES: List[OffenseTemplate] = [
    OffenseTemplate(
        offense_type=OffenseType.THEFT,
        raw_templates=[
            "আমার {item} চুরি {location} {time} {date}",
            "{item} হারিয়ে গেছে চুরি {location} {date} {time}",
            "কেউ আমার {item} নিয়ে গেছে {location} {time}",
            "{location} {date} {time} আমার {item} চুরি হয়েছে",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে আনুমানিক {time} সময়ে {location} এলাকায় অভিযোগকারীর "
            "{item} অজ্ঞাতনামা চোর/চোরেরা কৌশলে চুরি করে নিয়ে যায়। এই ঘটনায় অভিযোগকারী "
            "উক্ত চোরকে/চোরদের সনাক্ত করতে পারেননি। আইনানুগ ব্যবস্থা গ্রহণের জন্য এই অভিযোগ দাখিল করা হচ্ছে।"
        ),
        uses_item=True,
        property_field=lambda item: item,
    ),
    OffenseTemplate(
        offense_type=OffenseType.ROBBERY_SNATCHING,
        raw_templates=[
            "আমার {item} ছিনতাই {location} {time}",
            "রাস্তায় {item} কেড়ে নিয়ে গেছে {location} {date}",
            "{location} {date} {time} আমার {item} ছিনতাই হয়েছে",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {time} সময়ে অভিযোগকারী {location} এলাকায় রাস্তা দিয়ে "
            "যাওয়ার সময় অজ্ঞাতনামা দুর্বৃত্ত/দুর্বৃত্তরা অতর্কিতভাবে তার {item} জোরপূর্বক ছিনিয়ে "
            "নিয়ে দ্রুত পালিয়ে যায়। অভিযোগকারী চিৎকার করলেও আশপাশের কেউ ধরতে পারেনি।"
        ),
        uses_item=True,
        property_field=lambda item: item,
    ),
    OffenseTemplate(
        offense_type=OffenseType.DACOITY,
        raw_templates=[
            "কয়েকজন মিলে বাসায় ডাকাতি {location} {date} {time}",
            "রাতে দল বেঁধে ডাকাতি বাড়িতে {location}",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {time} সময়ে {location} এলাকায় অভিযোগকারীর বসতবাড়িতে "
            "{accused} সহ কয়েকজন দুর্বৃত্ত দলবদ্ধভাবে প্রবেশ করে ভয়ভীতি প্রদর্শন করে নগদ "
            "{amount} ও অন্যান্য মূল্যবান সামগ্রী লুট করে নিয়ে যায়।"
        ),
        uses_amount=True,
        uses_accused=True,
        property_field=lambda amount: f"নগদ {amount} ও মূল্যবান জিনিসপত্র",
    ),
    OffenseTemplate(
        offense_type=OffenseType.FRAUD,
        raw_templates=[
            "{accused} টাকা নিয়ে প্রতারণা {amount}",
            "অনলাইনে {amount} প্রতারণা করেছে {accused}",
            "{accused} ব্যবসার কথা বলে টাকা নিয়ে পালিয়ে গেছে {amount}",
        ],
        narrative_template=(
            "অভিযোগকারীর সাথে {accused} বিভিন্ন সময়ে ব্যবসায়িক/আর্থিক লেনদেনের কথা বলে "
            "বিশ্বাস অর্জন করে {amount} গ্রহণ করে। পরবর্তীতে {accused} উক্ত টাকা ফেরত না দিয়ে "
            "প্রতারণার আশ্রয় নেয় এবং যোগাযোগ বন্ধ করে দেয়।"
        ),
        uses_amount=True,
        uses_accused=True,
        property_field=lambda amount: f"প্রতারণামূলকভাবে গৃহীত {amount}",
    ),
    OffenseTemplate(
        offense_type=OffenseType.ASSAULT,
        raw_templates=[
            "{accused} আমাকে মারধর {location} {time}",
            "কেউ আমাকে মারল রাস্তায় {location} {date}",
            "{accused} মারধর করেছে আহত {location}",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {time} সময়ে {location} এলাকায় {accused} অভিযোগকারীর "
            "সাথে বাদানুবাদের জেরে অতর্কিতভাবে মারধর করে গুরুতর জখম করে। অভিযোগকারী "
            "চিকিৎসার জন্য নিকটস্থ হাসপাতালে ভর্তি হন।"
        ),
        uses_accused=True,
    ),
    OffenseTemplate(
        offense_type=OffenseType.SEXUAL_HARASSMENT,
        raw_templates=[
            "{accused} উত্ত্যক্ত করেছে রাস্তায় {location}",
            "স্কুলে যাওয়ার পথে উত্ত্যক্ত {location} {time}",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {time} সময়ে {location} এলাকায় {accused} অভিযোগকারীকে "
            "উদ্দেশ্যপ্রণোদিতভাবে অশ্লীল অঙ্গভঙ্গি ও কুরুচিপূর্ণ মন্তব্য করে উত্ত্যক্ত করে, যা "
            "যৌন নির্যাতনের শামিল।"
        ),
        uses_accused=True,
    ),
    OffenseTemplate(
        offense_type=OffenseType.KIDNAPPING,
        raw_templates=[
            "{victim} অপহরণ {location} {date}",
            "{victim} কে জোর করে তুলে নিয়ে গেছে {location}",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {location} এলাকা থেকে {victim} কে {accused} জোরপূর্বক "
            "একটি গাড়িতে তুলে নিয়ে যায়। এরপর থেকে {victim} এর কোনো সন্ধান পাওয়া যাচ্ছে না।"
        ),
        uses_victim=True,
        uses_accused=True,
    ),
    OffenseTemplate(
        offense_type=OffenseType.MISSING_PERSON,
        raw_templates=[
            "{victim} নিখোঁজ {date} {location}",
            "{victim} বাসা থেকে বের হয়ে ফিরেনি {date}",
        ],
        narrative_template=(
            "{victim} গত {date} তারিখে {location} এলাকার বাসা থেকে বের হওয়ার পর থেকে "
            "নিখোঁজ রয়েছে। বিভিন্ন স্থানে খোঁজাখুঁজি করেও তার কোনো সন্ধান পাওয়া যায়নি।"
        ),
        uses_victim=True,
    ),
    OffenseTemplate(
        offense_type=OffenseType.VEHICLE_THEFT,
        raw_templates=[
            "আমার {item} চুরি পার্কিং থেকে {location} {date}",
            "{item} চুরি হয়েছে {location} {time}",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {time} সময়ে {location} এলাকায় পার্কিং করে রাখা "
            "অভিযোগকারীর {item} অজ্ঞাতনামা ব্যক্তি/ব্যক্তিরা চুরি করে নিয়ে যায়।"
        ),
        uses_item=True,
        property_field=lambda item: item,
    ),
    OffenseTemplate(
        offense_type=OffenseType.CYBER_CRIME,
        raw_templates=[
            "ফেসবুক আইডি হ্যাক {date}",
            "বিকাশ থেকে টাকা প্রতারণা {amount} কল করে",
            "আমার আইডি হ্যাক করে টাকা চাচ্ছে বন্ধুদের কাছে",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে অজ্ঞাতনামা ব্যক্তি অভিযোগকারীর ফেসবুক/মোবাইল ব্যাংকিং "
            "অ্যাকাউন্ট অবৈধভাবে দখল করে এবং প্রতারণার মাধ্যমে {amount} আত্মসাৎ করে।"
        ),
        uses_amount=True,
        property_field=lambda amount: f"অনলাইন প্রতারণায় খোয়া যাওয়া {amount}",
    ),
    OffenseTemplate(
        offense_type=OffenseType.THREAT,
        raw_templates=[
            "{accused} হুমকি দিয়েছে জীবন নাশের {location}",
            "ফোনে হুমকি দিয়েছে {accused}",
        ],
        narrative_template=(
            "অদ্য {date} তারিখে {accused} অভিযোগকারীকে {location} এলাকায় প্রাণনাশের "
            "হুমকি প্রদান করে এবং ভবিষ্যতে গুরুতর ক্ষতি করার ভয়ভীতি দেখায়।"
        ),
        uses_accused=True,
    ),
    OffenseTemplate(
        offense_type=OffenseType.DOWRY_HARASSMENT,
        raw_templates=[
            "স্বামী যৌতুকের জন্য নির্যাতন করে {location}",
            "যৌতুকের জন্য মারধর করে শ্বশুরবাড়িতে {location}",
        ],
        narrative_template=(
            "অভিযোগকারীর {accused} বিবাহের পর থেকে অতিরিক্ত যৌতুকের দাবিতে {location} "
            "এলাকার বাসায় শারীরিক ও মানসিকভাবে নির্যাতন করে আসছে। অদ্য {date} তারিখে "
            "সর্বশেষ নির্যাতনের ঘটনা ঘটে।"
        ),
        uses_accused=True,
        accused_pool=HUSBAND_NAMES,
    ),
]


def fill(template: str, **kwargs) -> str:
    return template.format(**{k: v for k, v in kwargs.items() if v is not None})


def _placeholders(template: str) -> set:
    return set(re.findall(r"\{(\w+)\}", template))


def _ground_raw_text(raw_text: str, raw_template: str, tmpl: OffenseTemplate, slots: Dict) -> str:
    """Append any fact the narrative states but the chosen raw phrasing
    omitted, so every fact in ``target`` is traceable to ``raw_text``.
    """
    raw_placeholders = _placeholders(raw_template)
    narrative_placeholders = _placeholders(tmpl.narrative_template)

    for slot_name in _ALWAYS_GROUNDED_SLOTS:
        if slot_name not in raw_placeholders and slots.get(slot_name) is not None:
            raw_text = f"{raw_text} {slots[slot_name]}"

    for slot_name in _CONDITIONALLY_GROUNDED_SLOTS:
        if (
            slot_name in narrative_placeholders
            and slot_name not in raw_placeholders
            and slots.get(slot_name) is not None
        ):
            raw_text = f"{raw_text} {slots[slot_name]}"

    return raw_text


def build_example(rng: random.Random, tmpl: OffenseTemplate) -> Dict:
    location = rng.choice(LOCATIONS)
    time_ = rng.choice(TIMES)
    date_ = random_recent_date(rng)
    accused = rng.choice(tmpl.accused_pool or ACCUSED_NAMES) if tmpl.uses_accused else "অজ্ঞাত"
    complainant = rng.choice(COMPLAINANT_NAMES)
    victim = rng.choice(VICTIM_NAMES) if tmpl.uses_victim else complainant
    item = rng.choice(ITEMS) if tmpl.uses_item else None
    amount = rng.choice(AMOUNTS) if tmpl.uses_amount else None

    slots = dict(
        location=location,
        time=time_,
        date=date_,
        accused=accused,
        victim=victim,
        item=item,
        amount=amount,
    )

    raw_template = rng.choice(tmpl.raw_templates)
    raw_text = fill(raw_template, **slots)
    raw_text = _ground_raw_text(raw_text, raw_template, tmpl, slots)
    narrative = fill(tmpl.narrative_template, **slots)

    property_description = None
    if tmpl.property_field is not None:
        source = item if tmpl.uses_item else amount
        property_description = tmpl.property_field(source)

    witnesses = []
    if rng.random() < 0.35:
        witnesses = [rng.choice(WITNESS_NAME_POOL)]

    target = {
        "complainant_name": complainant,
        "complainant_address": maybe(rng, f"{location} (অভিযোগকারীর বসবাসের ঠিকানা)", p=0.5),
        "complainant_phone": maybe(rng, f"01{rng.randint(300000000, 999999999)}", p=0.4),
        "victim_name": victim if tmpl.uses_victim else None,
        "incident_date": date_,
        "incident_time": time_,
        "incident_location": location,
        "thana": THANA_BY_LOCATION.get(location),
        "offense_type": tmpl.offense_type.value,
        "accused_name": accused,
        "accused_description": None,
        "property_description": property_description,
        "witnesses": witnesses,
        "narrative": narrative,
    }

    return {"raw_text": raw_text, "target": target}


def generate(n_per_template: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    examples: List[Dict] = []
    for tmpl in TEMPLATES:
        for _ in range(n_per_template):
            examples.append(build_example(rng, tmpl))
    rng.shuffle(examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-template", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "processed", "synthetic_raw.jsonl"),
    )
    args = parser.parse_args()

    examples = generate(args.n_per_template, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} synthetic examples to {args.out}")


if __name__ == "__main__":
    main()
