#!/usr/bin/env python3
"""Generate a synthetic seed dataset for the Bangla-FIR LLM.

This is a *bootstrap* dataset meant to unblock development and let the
fine-tuning pipeline be built, tested, and smoke-tested end-to-end
before the real, human-curated dataset (1,000+ real FIR templates, per
the project's data-collection plan) is ready.

Each example pairs a short, informal Bangla sentence -- the kind of
text the ST-GNN sign-recognition stage would emit -- with the
structured FIR JSON object the LLM should learn to produce.

Usage:
    python scripts/generate_seed_dataset.py --out data/seed_dataset.jsonl --count 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

RNG_SEED = 42

# Each offense defines its own raw (informal) phrasing template so the
# generated sign-derived sentence stays consistent with the FIR label
# assigned to it, plus a formal description template for the FIR body.
# ``needs_item`` marks offenses whose templates reference {item}.
OFFENSES = [
    {
        "key": "চুরি",
        "needs_item": True,
        "raw_templates": ["আমার {item} চুরি হয়েছে {location} {time_short}"],
        "description": "অভিযোগকারীর {item} {location}-এ {time} সময়ে চুরি হয়ে যায়।",
    },
    {
        "key": "ছিনতাই",
        "needs_item": True,
        "raw_templates": ["{location} আমার {item} ছিনতাই {time_short}"],
        "description": "অভিযোগকারীর {item} {location}-এ {time} সময়ে জোরপূর্বক ছিনতাই করা হয়।",
    },
    {
        "key": "হারানো",
        "needs_item": True,
        "raw_templates": ["{location} আমি {item} হারিয়েছি {time_short}"],
        "description": "অভিযোগকারী {location}-এ {time} সময়ে তার {item} হারিয়ে ফেলেছেন।",
    },
    {
        "key": "প্রতারণা",
        "needs_item": False,
        "raw_templates": ["আমার সাথে প্রতারণা হয়েছে {location} {time_short}"],
        "description": "অভিযুক্ত ব্যক্তি {location}-এ {time} সময়ে অভিযোগকারীর কাছ থেকে প্রতারণার মাধ্যমে অর্থ আদায় করে।",
    },
    {
        "key": "মারামারি",
        "needs_item": False,
        "raw_templates": ["আমাকে {location} মারামারি {time_short}"],
        "description": "{location}-এ {time} সময়ে অভিযোগকারী শারীরিকভাবে আক্রান্ত হন।",
    },
    {
        "key": "হুমকি",
        "needs_item": False,
        "raw_templates": ["কেউ আমাকে হুমকি দিয়েছে {location} {time_short}"],
        "description": "অভিযুক্ত ব্যক্তি {location}-এ {time} সময়ে অভিযোগকারীকে প্রাণনাশের হুমকি দেয়।",
    },
    {
        "key": "যৌন হয়রানি",
        "needs_item": False,
        "raw_templates": ["{location} আমি হয়রানির শিকার হয়েছি {time_short}"],
        "description": "{location}-এ {time} সময়ে অভিযোগকারী যৌন হয়রানির শিকার হন।",
    },
    {
        "key": "গৃহে চুরি",
        "needs_item": False,
        "raw_templates": ["আমার বাসায় চুরি {location} {time_short}"],
        "description": "অভিযোগকারীর বাসায় {location}-এ {time} সময়ে অজ্ঞাত ব্যক্তি প্রবেশ করে মূল্যবান সামগ্রী নিয়ে যায়।",
    },
]

ITEMS = ["মানিব্যাগ", "মোবাইল ফোন", "ল্যাপটপ", "গহনা", "সাইকেল", "নগদ টাকা", "ব্যাগ", "দলিল"]

LOCATIONS = [
    "উত্তরা", "ধানমন্ডি", "মিরপুর", "গুলশান", "মোহাম্মদপুর", "মালিবাগ",
    "চট্টগ্রাম শহর", "রাজশাহী শহর", "সিলেট শহর", "খুলনা শহর", "বনানী", "যাত্রাবাড়ী",
]

TIMES = [
    ("সকাল ৮টা", "সকাল ৮টায়"), ("সকাল ১০টা", "সকাল ১০টায়"), ("দুপুর ১টা", "দুপুর ১টায়"),
    ("বিকাল ৫টা", "বিকাল ৫টায়"), ("সন্ধ্যা ৭টা", "সন্ধ্যা ৭টায়"), ("রাত ৯টা", "রাত ৯টায়"),
    ("রাত ১১টা", "রাত ১১টায়"),
]

DATES = [
    "০১/০৬/২০২৬", "০৫/০৬/২০২৬", "১২/০৬/২০২৬", "১৮/০৬/২০২৬", "২২/০৬/২০২৬",
    "২৭/০৬/২০২৬", "৩০/০৬/২০২৬", "০৩/০৫/২০২৬", "১৫/০৫/২০২৬", "২৯/০৫/২০২৬",
]

FIRST_NAMES = ["রহিম", "করিম", "সালমা", "ফাতেমা", "জামাল", "আয়েশা", "নাসির", "রুমা", "সোহেল", "মিতু"]
LAST_NAMES = ["ইসলাম", "হোসেন", "আক্তার", "বেগম", "খান", "আহমেদ", "চৌধুরী", "মিয়া"]


def _contact() -> str:
    return f"01{random.randint(300000000, 999999999)}"


def make_example() -> Dict[str, str]:
    offense = random.choice(OFFENSES)
    item = random.choice(ITEMS) if offense["needs_item"] else None
    location = random.choice(LOCATIONS)
    time_short, time_long = random.choice(TIMES)
    date = random.choice(DATES)
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

    raw_template = random.choice(offense["raw_templates"])
    raw_text = raw_template.format(item=item, location=location, time_short=time_short)
    description = offense["description"].format(item=item, location=location, time=time_long)

    fir = {
        "date_of_occurrence": date,
        "time_of_occurrence": time_long,
        "location": f"{location}, বাংলাদেশ",
        "offense_type": offense["key"],
        "complainant_name": name,
        "complainant_address": f"{location}, বাংলাদেশ",
        "complainant_contact": _contact(),
        "accused": "অজ্ঞাত",
        "description": description,
    }
    return {"input_text": raw_text, "fir": fir}


def generate(count: int) -> List[Dict[str, str]]:
    seen = set()
    examples = []
    attempts = 0
    while len(examples) < count and attempts < count * 20:
        attempts += 1
        ex = make_example()
        key = (ex["input_text"], ex["fir"]["date_of_occurrence"], ex["fir"]["complainant_name"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "seed_dataset.jsonl")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    examples = generate(args.count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} synthetic examples to {args.out}")


if __name__ == "__main__":
    main()
