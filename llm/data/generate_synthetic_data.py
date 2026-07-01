#!/usr/bin/env python3
"""Generate a synthetic BdSL-raw-text -> FIR-JSON dataset for bootstrapping
the LLM fine-tuning pipeline.

This is a *starter* dataset built entirely offline from hand-written Bangla
templates (see ``offense_catalog.py`` / ``value_banks.py``) so the training
pipeline can be developed, tested, and smoke-tested without waiting on the
real ~1,000+ FIR templates that the project plans to collect from BLAST /
police sources. Once real (raw sign-text, FIR) pairs are available, feed them
through the same JSON schema using ``prepare_dataset.py`` and mix them in with
(or replace) this synthetic data before a production fine-tuning run.

Usage:
    python generate_synthetic_data.py --n 800 --seed 42 --out-dir processed
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offense_catalog import OFFENSE_TYPES, OffenseType  # noqa: E402
from value_banks import (  # noqa: E402
    ALL_DATE_PHRASES,
    ALL_NAMES,
    DAMAGED_PROPERTY_ITEMS,
    DATE_PHRASES_FIXED_OFFSET,
    DISTRICTS,
    RELATION_PHRASES_KNOWN_SUSPECT,
    STOLEN_ITEMS,
    THANAS,
    TIME_PHRASES,
    WEEKDAY_PHRASES,
)

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def to_bn_digits(s: str) -> str:
    return s.translate(BN_DIGITS)


def format_date_bn(d: date) -> str:
    return to_bn_digits(d.strftime("%d/%m/%Y"))


def resolve_occurrence_date(report_date: date, date_phrase: str) -> date:
    if date_phrase in DATE_PHRASES_FIXED_OFFSET:
        return report_date + timedelta(days=DATE_PHRASES_FIXED_OFFSET[date_phrase])
    if date_phrase in WEEKDAY_PHRASES:
        target_weekday = WEEKDAY_PHRASES[date_phrase]
        days_back = (report_date.weekday() - target_weekday) % 7
        days_back = days_back or 7  # "গত <বার>" always refers to a past day
        return report_date - timedelta(days=days_back)
    raise ValueError(f"Unknown date phrase: {date_phrase}")


def pick_suspect(rng: random.Random, victim_name: str) -> tuple[str, str]:
    """Return (suspect_phrase_bn, accused_name) for the narrative/JSON."""
    if rng.random() < 0.55:
        return "একজন অজ্ঞাতনামা ব্যক্তি", "অজ্ঞাতনামা"
    relation = rng.choice(RELATION_PHRASES_KNOWN_SUSPECT)
    candidates = [n for n in ALL_NAMES if n != victim_name]
    suspect_name = rng.choice(candidates)
    return f"তার {relation} {suspect_name}", suspect_name


def pick_item(rng: random.Random, offense: OffenseType) -> str | None:
    if not offense.needs_item:
        return None
    bank = STOLEN_ITEMS if offense.item_bank == "stolen" else DAMAGED_PROPERTY_ITEMS
    return rng.choice(bank)


def make_record(rng: random.Random, idx: int, offense: OffenseType) -> dict:
    report_date = date.today() - timedelta(days=rng.randint(0, 365))
    date_phrase = rng.choice(ALL_DATE_PHRASES)
    occurrence_date = resolve_occurrence_date(report_date, date_phrase)
    time_phrase = rng.choice(list(TIME_PHRASES))
    time_24h = TIME_PHRASES[time_phrase]

    location = rng.choice(THANAS)
    district = "ঢাকা" if rng.random() < 0.85 else rng.choice(DISTRICTS)

    complainant_name = rng.choice(ALL_NAMES)
    complainant_phone = "01" + "".join(str(rng.randint(0, 9)) for _ in range(9))
    complainant_address = f"{location}, {district}"

    if offense.is_missing_person:
        victim_name = rng.choice([n for n in ALL_NAMES if n != complainant_name])
        accused_name = "প্রযোজ্য নয়"
        suspect_phrase = ""
        item = None
    else:
        victim_name = complainant_name
        suspect_phrase, accused_name = pick_suspect(rng, victim_name)
        item = pick_item(rng, offense)

    fmt_kwargs = {
        "victim": victim_name,
        "location": location,
        "time": time_phrase,
        "date": date_phrase,
        "item": item or "",
        "suspect_phrase": suspect_phrase,
    }

    raw_template = rng.choice(offense.raw_templates)
    raw_signed_text = raw_template.format(**fmt_kwargs).strip()
    raw_signed_text = " ".join(raw_signed_text.split())  # collapse extra spaces

    narrative_kwargs = dict(fmt_kwargs)
    narrative_kwargs["date"] = format_date_bn(occurrence_date)
    narrative_bn = offense.narrative_template.format(**narrative_kwargs)
    narrative_bn = " ".join(narrative_bn.split())

    fir = {
        "thana": location,
        "district": district,
        "date_of_occurrence": occurrence_date.isoformat(),
        "time_of_occurrence": time_24h,
        "date_of_report": report_date.isoformat(),
        "place_of_occurrence": f"{location} এলাকা",
        "complainant_name": complainant_name,
        "complainant_address": complainant_address,
        "complainant_phone": complainant_phone,
        "victim_name": victim_name,
        "accused_name": accused_name,
        "offense_type": offense.name_bn,
        "penal_code_sections": list(offense.sections),
        "items_involved": [item] if item else [],
        "narrative_bn": narrative_bn,
    }

    return {
        "id": f"SYN-{idx:06d}",
        "raw_signed_text": raw_signed_text,
        "offense_type_key": offense.key,
        "fir": fir,
        "target_json": json.dumps(fir, ensure_ascii=False),
    }


def generate_dataset(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        offense = rng.choice(OFFENSE_TYPES)
        records.append(make_record(rng, i, offense))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800, help="Total number of records to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Output directory for train.jsonl / val.jsonl / test.jsonl",
    )
    args = parser.parse_args()

    records = generate_dataset(args.n, args.seed)

    n_train = int(len(records) * args.train_frac)
    n_val = int(len(records) * args.val_frac)

    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]

    write_jsonl(train, args.out_dir / "train.jsonl")
    write_jsonl(val, args.out_dir / "val.jsonl")
    write_jsonl(test, args.out_dir / "test.jsonl")

    print(f"Wrote {len(train)} train / {len(val)} val / {len(test)} test records to {args.out_dir}")


if __name__ == "__main__":
    main()
