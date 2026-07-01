"""Synthetic Bangla FIR dataset generator.

The real project fine-tunes on a corpus of 1,000+ collected FIR templates
(see the development pipeline). Until that proprietary corpus is available,
this module programmatically synthesises realistic ``(raw_statement, FIR)``
pairs so the full training + inference pipeline can be developed, tested and
demonstrated end to end.

Each example contains:

* ``raw_statement`` — an informal, terse Bangla sentence resembling the output
  of the BdSL -> text recognition stage.
* ``complaint``     — a :class:`FIRComplaint` (structured, formal FIR).
* ``extraction``    — the ground-truth :class:`FIRExtraction` entities.

The generator is fully deterministic given a seed so datasets are reproducible.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .schema import FIRComplaint, FIRExtraction

UNKNOWN = "অজ্ঞাত"

# --- Vocabulary pools --------------------------------------------------------

MALE_NAMES = [
    "রহিম উদ্দিন", "করিম মিয়া", "আব্দুল কাদের", "সজীব হোসেন", "রফিকুল ইসলাম",
    "জাহিদ হাসান", "মামুন অর রশিদ", "নাসির আহমেদ", "শাকিল আহমেদ", "তানভীর হোসেন",
]
FEMALE_NAMES = [
    "রহিমা বেগম", "ফাতেমা আক্তার", "নাসরিন সুলতানা", "সালমা খাতুন", "আয়েশা সিদ্দিকা",
    "শারমিন আক্তার", "রুবিনা ইয়াসমিন", "মৌসুমী আক্তার", "তাসলিমা বেগম", "লাবণী আক্তার",
]
LOCATIONS = [
    "উত্তরা", "মিরপুর ১০", "ধানমন্ডি ২৭", "গুলশান ১", "মতিঝিল", "বাড্ডা",
    "মোহাম্মদপুর", "যাত্রাবাড়ী", "খিলগাঁও", "বনানী", "শাহবাগ", "ফার্মগেট",
    "চট্টগ্রাম আগ্রাবাদ", "সিলেট জিন্দাবাজার", "রাজশাহী সাহেব বাজার", "খুলনা সোনাডাঙ্গা",
]
TIMES = [
    "সকাল ৮টা", "সকাল ১০টা", "দুপুর ১২টা", "দুপুর ২টা", "বিকেল ৪টা", "বিকেল ৫টা",
    "সন্ধ্যা ৬টা", "সন্ধ্যা ৭টা", "রাত ৯টা", "রাত ১১টা", "গভীর রাত",
]
VALUABLES = [
    "মানিব্যাগ", "মোবাইল ফোন", "স্বর্ণের চেইন", "ল্যাপটপ", "নগদ টাকা",
    "হাতঘড়ি", "ব্যাগ", "মোটরসাইকেল", "সাইকেল", "স্বর্ণের আংটি",
]

BN_DIGITS = {"0": "০", "1": "১", "2": "২", "3": "৩", "4": "৪",
             "5": "৫", "6": "৬", "7": "৭", "8": "৮", "9": "৯"}
BN_MONTHS = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]


def _to_bn_digits(value: int) -> str:
    return "".join(BN_DIGITS.get(ch, ch) for ch in str(value))


def _random_date(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    month = rng.choice(BN_MONTHS)
    year = rng.choice([2024, 2025, 2026])
    return f"{_to_bn_digits(day)} {month} {_to_bn_digits(year)}"


# --- Offense definitions -----------------------------------------------------
# Each builder returns (raw_statement, extraction) given random context.


def _theft(rng, victim, accused, location, date, time) -> tuple[str, FIRExtraction]:
    item = rng.choice(VALUABLES)
    raw_templates = [
        f"আমার {item} চুরি হয়েছে {location} {time}",
        f"{location} থেকে আমার {item} চুরি {time}",
        f"{time} {location} আমার {item} চুরি হয়ে গেছে",
    ]
    ext = FIRExtraction(
        complainant_name=victim, victim_name=victim, accused_name=UNKNOWN,
        offense_type="চুরি", location=location, incident_date=date,
        incident_time=time, stolen_items=[item],
        description=f"{location} এলাকায় {item} চুরির ঘটনা",
    )
    return rng.choice(raw_templates), ext


def _mugging(rng, victim, accused, location, date, time) -> tuple[str, FIRExtraction]:
    item = rng.choice(VALUABLES)
    raw_templates = [
        f"{location} {time} আমার {item} ছিনতাই হয়েছে",
        f"আমার {item} ছিনতাই {location} {time}",
        f"{time} {location} দুইজন লোক আমার {item} ছিনিয়ে নিয়েছে",
    ]
    ext = FIRExtraction(
        complainant_name=victim, victim_name=victim,
        accused_name="অজ্ঞাতনামা ছিনতাইকারী", offense_type="ছিনতাই",
        location=location, incident_date=date, incident_time=time,
        stolen_items=[item],
        description=f"{location} এলাকায় {item} ছিনতাইয়ের ঘটনা",
    )
    return rng.choice(raw_templates), ext


def _assault(rng, victim, accused, location, date, time) -> tuple[str, FIRExtraction]:
    raw_templates = [
        f"{accused} আমাকে মারধর করেছে {location} {time}",
        f"{location} {time} {accused} আমার উপর হামলা করেছে",
        f"{time} {location} আমাকে মেরেছে {accused}",
    ]
    ext = FIRExtraction(
        complainant_name=victim, victim_name=victim, accused_name=accused,
        offense_type="মারধর / হামলা", location=location, incident_date=date,
        incident_time=time, stolen_items=[],
        description=f"{location} এলাকায় {victim}-কে মারধরের ঘটনা",
    )
    return rng.choice(raw_templates), ext


def _threat(rng, victim, accused, location, date, time) -> tuple[str, FIRExtraction]:
    raw_templates = [
        f"{accused} আমাকে হুমকি দিয়েছে {location} {time}",
        f"{location} {time} {accused} আমাকে প্রাণনাশের হুমকি দিয়েছে",
    ]
    ext = FIRExtraction(
        complainant_name=victim, victim_name=victim, accused_name=accused,
        offense_type="হুমকি প্রদান", location=location, incident_date=date,
        incident_time=time, stolen_items=[],
        description=f"{accused} কর্তৃক {victim}-কে হুমকির ঘটনা",
    )
    return rng.choice(raw_templates), ext


def _fraud(rng, victim, accused, location, date, time) -> tuple[str, FIRExtraction]:
    amount = f"{_to_bn_digits(rng.choice([5, 10, 20, 50]))} হাজার টাকা"
    raw_templates = [
        f"{accused} আমার {amount} প্রতারণা করে নিয়েছে {time}",
        f"{time} {accused} প্রতারণার মাধ্যমে আমার {amount} নিয়েছে {location}",
    ]
    ext = FIRExtraction(
        complainant_name=victim, victim_name=victim, accused_name=accused,
        offense_type="প্রতারণা", location=location, incident_date=date,
        incident_time=time, stolen_items=[amount],
        description=f"{accused} কর্তৃক {amount} প্রতারণার ঘটনা",
    )
    return rng.choice(raw_templates), ext


OFFENSE_BUILDERS = [_theft, _mugging, _assault, _threat, _fraud]


def _build_complaint_body(ext: FIRExtraction) -> str:
    """Compose a formal Bangla FIR narrative from extracted entities."""
    items = "、".join(ext.stolen_items) if ext.stolen_items else ""
    parts = [
        f"আমি, {ext.complainant_name}, এই মর্মে অভিযোগ দায়ের করছি যে "
        f"গত {ext.incident_date} তারিখ আনুমানিক {ext.incident_time} সময়ে "
        f"{ext.location} এলাকায় একটি {ext.offense_type} সংক্রান্ত ঘটনা সংঘটিত হয়।",
    ]
    if ext.accused_name and ext.accused_name != UNKNOWN:
        parts.append(f"এ ঘটনায় {ext.accused_name} জড়িত বলে আমি অভিযোগ করছি।")
    else:
        parts.append("ঘটনার সাথে জড়িত ব্যক্তি অজ্ঞাত।")
    if items:
        parts.append(f"উক্ত ঘটনায় আমার {items} খোয়া/ক্ষতিগ্রস্ত হয়।")
    parts.append(
        "অতএব, উপরোক্ত ঘটনার সুষ্ঠু তদন্ত করে দোষী ব্যক্তির বিরুদ্ধে "
        "আইনানুগ ব্যবস্থা গ্রহণের জন্য বিনীত অনুরোধ জানাচ্ছি।"
    )
    return " ".join(parts)


def _extraction_to_complaint(ext: FIRExtraction) -> FIRComplaint:
    return FIRComplaint(
        offense_type=ext.offense_type or UNKNOWN,
        incident_date=ext.incident_date or UNKNOWN,
        incident_time=ext.incident_time or UNKNOWN,
        location=ext.location or UNKNOWN,
        complainant_name=ext.complainant_name or UNKNOWN,
        victim_name=ext.victim_name or UNKNOWN,
        accused_name=ext.accused_name or UNKNOWN,
        stolen_items=list(ext.stolen_items),
        complaint_body=_build_complaint_body(ext),
    )


def generate_example(rng: random.Random) -> Dict:
    """Generate a single synthetic training example."""
    victim = rng.choice(MALE_NAMES + FEMALE_NAMES)
    accused = rng.choice(MALE_NAMES + FEMALE_NAMES + ["অজ্ঞাতনামা ব্যক্তি"])
    location = rng.choice(LOCATIONS)
    date = _random_date(rng)
    time = rng.choice(TIMES)

    builder = rng.choice(OFFENSE_BUILDERS)
    raw_statement, ext = builder(rng, victim, accused, location, date, time)
    complaint = _extraction_to_complaint(ext)

    return {
        "raw_statement": raw_statement,
        "extraction": ext.model_dump(),
        "complaint": complaint.model_dump(),
    }


def generate_dataset(num_examples: int, seed: int = 42) -> List[Dict]:
    """Generate a list of ``num_examples`` deterministic synthetic examples."""
    rng = random.Random(seed)
    seen: set[str] = set()
    examples: List[Dict] = []
    # Oversample and de-duplicate raw statements to keep the set diverse.
    attempts = 0
    while len(examples) < num_examples and attempts < num_examples * 20:
        attempts += 1
        ex = generate_example(rng)
        key = ex["raw_statement"]
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)
    return examples
