"""Synthetic bootstrap-dataset generator for the FIR-generation LLM.

Real training data for this stage should ultimately come from ~1,000+ real
FIR templates collected from BLAST / police records (see repo README /
workflow table). Since that corpus is still being collected, this generator
produces a large, diverse set of *synthetic* (informal Bangla sign-derived
text -> structured FIR JSON) pairs so that:

1. The full training/inference/evaluation pipeline can be built, tested and
   demoed end-to-end today.
2. Real FIR templates can later be dropped into
   ``data/templates/real_examples.jsonl`` and mixed in via
   ``build_dataset(..., extra_jsonl_paths=[...])`` without changing any code.

Usage
-----
    python -m src.data_generation --n 1200 --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from .schema import (
    DEFAULT_PENAL_CODE_SECTIONS,
    FIRRecord,
    OffenseType,
    Person,
    PropertyItem,
)

NAMES = [
    "করিম উদ্দিন", "রহিমা বেগম", "সাকিব হাসান", "নাজমুল ইসলাম", "ফারজানা আক্তার",
    "আব্দুল কাদের", "সুমাইয়া ইসলাম", "তানভীর আহমেদ", "মোছাম্মৎ সালমা", "জাহিদ হোসেন",
    "রাশেদা খাতুন", "ইমরান হোসেন", "নুসরাত জাহান", "মিজানুর রহমান", "শাহনাজ পারভীন",
    "আরিফুল ইসলাম", "তাসলিমা বেগম", "হাবিবুর রহমান", "রুমানা আক্তার", "শফিকুল ইসলাম",
]

LOCATIONS = [
    ("উত্তরা", "উত্তরা পশ্চিম থানা", "ঢাকা"),
    ("ধানমন্ডি", "ধানমন্ডি থানা", "ঢাকা"),
    ("মিরপুর", "মিরপুর থানা", "ঢাকা"),
    ("গুলশান", "গুলশান থানা", "ঢাকা"),
    ("বাড্ডা", "বাড্ডা থানা", "ঢাকা"),
    ("মোহাম্মদপুর", "মোহাম্মদপুর থানা", "ঢাকা"),
    ("আগ্রাবাদ", "আগ্রাবাদ থানা", "চট্টগ্রাম"),
    ("জিন্দাবাজার", "কোতোয়ালী থানা", "সিলেট"),
    ("শাহমখদুম", "রাজপাড়া থানা", "রাজশাহী"),
    ("বোয়ালিয়া", "বোয়ালিয়া থানা", "রাজশাহী"),
    ("খুলনা সদর", "খুলনা সদর থানা", "খুলনা"),
    ("বরিশাল সদর", "কোতোয়ালী থানা", "বরিশাল"),
]

TIME_PHRASES = [
    "সকাল ৮টা", "সকাল ১০টা", "দুপুর ১টা", "দুপুর ২টা", "বিকাল ৪টা",
    "বিকাল ৫টা", "সন্ধ্যা ৬টা", "সন্ধ্যা ৭টা", "রাত ৯টা", "রাত ১০টা", "রাত ১১টা",
]

ITEMS = [
    ("মানিব্যাগ", (500, 5000)),
    ("মোবাইল ফোন", (5000, 60000)),
    ("ল্যাপটপ", (30000, 120000)),
    ("সাইকেল", (3000, 15000)),
    ("মোটরসাইকেল", (80000, 250000)),
    ("ব্যাগ", (500, 8000)),
    ("স্বর্ণের গয়না", (20000, 300000)),
    ("নগদ টাকা", (1000, 100000)),
    ("ঘড়ি", (1000, 20000)),
]

# Each template maps to (informal_variants, narrative_builder)
def _theft_case(rng: random.Random) -> Dict:
    name = rng.choice(NAMES)
    loc, ps, district = rng.choice(LOCATIONS)
    time_ph = rng.choice(TIME_PHRASES)
    item_name, value_range = rng.choice(ITEMS)
    value = rng.randint(*value_range)
    date_str = _random_recent_date(rng)

    informal_variants = [
        f"আমার {item_name} {loc} চুরি {time_ph}",
        f"{name} এর {item_name} {loc} এ চুরি হয়েছে {time_ph}",
        f"{item_name} হারিয়ে গেছে {loc} {time_ph} চুরি",
    ]
    informal = rng.choice(informal_variants)

    narrative = (
        f"অভিযোগকারী {name} জানান যে, গত {date_str} তারিখে {time_ph} সময়ে "
        f"{loc} এলাকায় অবস্থানকালে অজ্ঞাতনামা চোর/চোরেরা তাঁর {item_name} "
        f"চুরি করে নিয়ে যায়, যার আনুমানিক মূল্য {value:,} টাকা। ঘটনার পর তিনি "
        f"আশেপাশে খোঁজাখুঁজি করেও কোনো সন্ধান পাননি। এমতাবস্থায় তিনি বিষয়টি "
        f"আইনানুগ ব্যবস্থা গ্রহণের জন্য থানায় অভিযোগ দাখিল করছেন।"
    )

    record = FIRRecord(
        offense_type=OffenseType.THEFT,
        complainant=Person(name=name, address=f"{loc}, {district}"),
        accused=None,
        accused_unknown=True,
        incident_date=date_str,
        incident_time=time_ph,
        incident_location=loc,
        police_station=ps,
        district=district,
        stolen_or_damaged_items=[
            PropertyItem(description=item_name, estimated_value_bdt=value, quantity=1)
        ],
        narrative_bn=narrative,
    )
    return {"informal": informal, "record": record}


def _robbery_case(rng: random.Random) -> Dict:
    name = rng.choice(NAMES)
    loc, ps, district = rng.choice(LOCATIONS)
    time_ph = rng.choice(TIME_PHRASES)
    item_name, value_range = rng.choice(ITEMS)
    value = rng.randint(*value_range)
    date_str = _random_recent_date(rng)

    informal_variants = [
        f"{name} কে {loc} এ ছিনতাই {time_ph}, {item_name} নিয়ে গেছে",
        f"আমাকে {loc} এ ছিনতাইকারী ধরেছে {time_ph} {item_name} কেড়ে নিয়েছে",
    ]
    informal = rng.choice(informal_variants)

    narrative = (
        f"অভিযোগকারী {name} জানান যে, গত {date_str} তারিখে {time_ph} সময়ে "
        f"{loc} এলাকায় ২/৩ জন অজ্ঞাতনামা দুর্বৃত্ত তাঁর গতিরোধ করে ভয়ভীতি প্রদর্শন করে "
        f"জোরপূর্বক তাঁর {item_name} (আনুমানিক মূল্য {value:,} টাকা) ছিনিয়ে নিয়ে "
        f"পালিয়ে যায়। এ ঘটনায় তিনি শারীরিক ও মানসিকভাবে আতঙ্কিত হয়ে পড়েন এবং "
        f"অভিযুক্তদের বিরুদ্ধে আইনানুগ ব্যবস্থা গ্রহণের আবেদন জানাচ্ছেন।"
    )

    record = FIRRecord(
        offense_type=OffenseType.ROBBERY,
        complainant=Person(name=name, address=f"{loc}, {district}"),
        accused=None,
        accused_unknown=True,
        incident_date=date_str,
        incident_time=time_ph,
        incident_location=loc,
        police_station=ps,
        district=district,
        stolen_or_damaged_items=[
            PropertyItem(description=item_name, estimated_value_bdt=value, quantity=1)
        ],
        narrative_bn=narrative,
    )
    return {"informal": informal, "record": record}


def _assault_case(rng: random.Random) -> Dict:
    name = rng.choice(NAMES)
    accused_name = rng.choice([n for n in NAMES if n != name])
    loc, ps, district = rng.choice(LOCATIONS)
    time_ph = rng.choice(TIME_PHRASES)
    date_str = _random_recent_date(rng)
    known_accused = rng.random() < 0.6

    informal_variants = [
        f"{name} কে {loc} এ মারধর {time_ph}",
        f"আমাকে {accused_name} মারধর করেছে {loc} {time_ph}" if known_accused else f"অজ্ঞাত লোক আমাকে মেরেছে {loc} {time_ph}",
    ]
    informal = rng.choice(informal_variants)

    accused_clause = (
        f"পরিচিত ব্যক্তি {accused_name}" if known_accused else "অজ্ঞাতনামা এক/একাধিক ব্যক্তি"
    )
    narrative = (
        f"অভিযোগকারী {name} জানান যে, গত {date_str} তারিখে {time_ph} সময়ে "
        f"{loc} এলাকায় {accused_clause} তাঁর সাথে বাকবিতণ্ডার জেরে অতর্কিতভাবে "
        f"শারীরিক আক্রমণ করে, যাতে তিনি শরীরে আঘাত প্রাপ্त হন। তিনি চিকিৎসা "
        f"গ্রহণ করেছেন এবং অভিযুক্তের বিরুদ্ধে আইনানুগ ব্যবস্থা গ্রহণের আবেদন জানান।"
    )

    record = FIRRecord(
        offense_type=OffenseType.ASSAULT,
        complainant=Person(name=name, address=f"{loc}, {district}"),
        accused=None if not known_accused else Person(name=accused_name),
        accused_unknown=not known_accused,
        incident_date=date_str,
        incident_time=time_ph,
        incident_location=loc,
        police_station=ps,
        district=district,
        narrative_bn=narrative,
    )
    return {"informal": informal, "record": record}


def _fraud_case(rng: random.Random) -> Dict:
    name = rng.choice(NAMES)
    accused_name = rng.choice([n for n in NAMES if n != name])
    loc, ps, district = rng.choice(LOCATIONS)
    date_str = _random_recent_date(rng)
    amount = rng.randint(5000, 500000)

    informal_variants = [
        f"{name} এর কাছ থেকে {accused_name} টাকা প্রতারণা করেছে {loc}",
        f"আমার টাকা প্রতারণা করে নিয়েছে {accused_name} {loc} এ",
    ]
    informal = rng.choice(informal_variants)

    narrative = (
        f"অভিযোগকারী {name} জানান যে, {accused_name} নামক ব্যক্তি গত {date_str} "
        f"তারিখে {loc} এলাকায় মিথ্যা প্রতিশ্রুতি দিয়ে প্রতারণামূলকভাবে তাঁর কাছ থেকে "
        f"{amount:,} টাকা গ্রহণ করে এবং পরবর্তীতে তা ফেরত দিতে অস্বীকার করে। "
        f"অভিযোগকারী উক্ত ব্যক্তির বিরুদ্ধে প্রতারণার অভিযোগে আইনানুগ ব্যবস্থা "
        f"গ্রহণের আবেদন জানাচ্ছেন।"
    )

    record = FIRRecord(
        offense_type=OffenseType.FRAUD,
        complainant=Person(name=name, address=f"{loc}, {district}"),
        accused=Person(name=accused_name),
        accused_unknown=False,
        incident_date=date_str,
        incident_location=loc,
        police_station=ps,
        district=district,
        stolen_or_damaged_items=[
            PropertyItem(description="নগদ টাকা", estimated_value_bdt=amount, quantity=1)
        ],
        narrative_bn=narrative,
    )
    return {"informal": informal, "record": record}


def _missing_person_case(rng: random.Random) -> Dict:
    victim_name = rng.choice(NAMES)
    reporter_name = rng.choice([n for n in NAMES if n != victim_name])
    loc, ps, district = rng.choice(LOCATIONS)
    date_str = _random_recent_date(rng)

    informal_variants = [
        f"{victim_name} {loc} থেকে নিখোঁজ {date_str} থেকে",
        f"আমার পরিবারের সদস্য {victim_name} খুঁজে পাওয়া যাচ্ছে না {loc}",
    ]
    informal = rng.choice(informal_variants)

    narrative = (
        f"অভিযোগকারী {reporter_name} জানান যে, তাঁর পরিবারের সদস্য {victim_name} "
        f"গত {date_str} তারিখ থেকে {loc} এলাকা থেকে নিখোঁজ রয়েছেন। বিভিন্নভাবে "
        f"খোঁজাখুঁজি করেও তাঁর কোনো সন্ধান পাওয়া যায়নি। এমতাবস্থায় নিখোঁজ ব্যক্তির "
        f"সন্ধান লাভে প্রয়োজনীয় ব্যবস্থা গ্রহণের জন্য আবেদন জানানো হলো।"
    )

    record = FIRRecord(
        offense_type=OffenseType.MISSING_PERSON,
        complainant=Person(name=reporter_name, address=f"{loc}, {district}"),
        victim=Person(name=victim_name, address=f"{loc}, {district}"),
        accused=None,
        accused_unknown=True,
        incident_date=date_str,
        incident_location=loc,
        police_station=ps,
        district=district,
        narrative_bn=narrative,
    )
    return {"informal": informal, "record": record}


def _harassment_case(rng: random.Random) -> Dict:
    name = rng.choice(NAMES)
    loc, ps, district = rng.choice(LOCATIONS)
    time_ph = rng.choice(TIME_PHRASES)
    date_str = _random_recent_date(rng)

    informal = f"{name} কে {loc} এ উত্তক্ত করেছে {time_ph}"

    narrative = (
        f"অভিযোগকারী {name} জানান যে, গত {date_str} তারিখে {time_ph} সময়ে "
        f"{loc} এলাকায় অজ্ঞাতনামা এক ব্যক্তি তাঁকে উত্তক্ত/উক্তি করে মানসিকভাবে "
        f"বিব্রত করে। এ ঘটনায় তিনি আতঙ্কিত ও অসম্মানিত হয়েছেন এবং অভিযুক্তের "
        f"বিরুদ্ধে আইনানুগ ব্যবস্থা গ্রহণের আবেদন জানাচ্ছেন।"
    )

    record = FIRRecord(
        offense_type=OffenseType.HARASSMENT,
        complainant=Person(name=name, address=f"{loc}, {district}"),
        accused=None,
        accused_unknown=True,
        incident_date=date_str,
        incident_time=time_ph,
        incident_location=loc,
        police_station=ps,
        district=district,
        narrative_bn=narrative,
    )
    return {"informal": informal, "record": record}


_GENERATORS = [
    _theft_case,
    _robbery_case,
    _assault_case,
    _fraud_case,
    _missing_person_case,
    _harassment_case,
]


def _random_recent_date(rng: random.Random, days_back: int = 90) -> str:
    import datetime

    d = datetime.date.today() - datetime.timedelta(days=rng.randint(0, days_back))
    return d.isoformat()


def generate_examples(n: int, seed: int = 42) -> List[Dict]:
    """Generate ``n`` synthetic (informal_text, FIRRecord) example dicts."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        gen = rng.choice(_GENERATORS)
        case = gen(rng)
        record: FIRRecord = case["record"].with_defaults_filled()
        record.raw_input_text = case["informal"]
        examples.append(
            {
                "input_text": case["informal"],
                "output_json": json.loads(record.model_dump_json()),
            }
        )
    return examples


def build_dataset(
    n: int,
    out_dir: Path,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    extra_jsonl_paths: List[Path] | None = None,
) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = generate_examples(n, seed=seed)

    for p in extra_jsonl_paths or []:
        p = Path(p)
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(examples)

    n_total = len(examples)
    n_val = int(n_total * val_frac)
    n_test = int(n_total * test_frac)
    n_train = n_total - n_val - n_test

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }

    counts = {}
    for split_name, rows in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split_name] = len(rows)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1200, help="Number of synthetic examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "processed"
    )
    parser.add_argument(
        "--extra-jsonl",
        type=Path,
        nargs="*",
        default=None,
        help="Optional path(s) to real, human-collected FIR jsonl files to mix in",
    )
    args = parser.parse_args()

    counts = build_dataset(args.n, args.out_dir, seed=args.seed, extra_jsonl_paths=args.extra_jsonl)
    print(f"Wrote dataset splits to {args.out_dir}: {counts}")


if __name__ == "__main__":
    main()
