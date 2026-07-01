"""
Synthetic FIR (First Information Report) data generator for Bangla legal text.

Generates training pairs:
  - informal_text: raw Bangla text from sign language recognition output
  - formal_fir:    structured FIR legal complaint in Bangla
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Bangla vocabulary pools
# ---------------------------------------------------------------------------

VICTIM_NAMES = [
    "রহিম উদ্দিন", "করিম মিয়া", "ফাতেমা বেগম", "সুমাইয়া আক্তার",
    "মোহাম্মদ আলী", "নুরুন্নাহার", "আবুল হাসান", "মরিয়ম বেগম",
    "জহির উদ্দিন", "রোকেয়া খানম", "মনির হোসেন", "সালমা বেগম",
]

SUSPECT_NAMES = [
    "কালাম শেখ", "বাদল মিয়া", "রফিক উদ্দিন", "জামাল হোসেন",
    "সাইফুল ইসলাম", "তোফাজ্জল হোসেন", "মিজান সরকার", "হানিফ মোল্লা",
    "আজিজুল হক", "ইব্রাহিম খান", "নাসির উদ্দিন", "সিরাজুল ইসলাম",
]

LOCATIONS = [
    "উত্তরা ৫নং সেক্টর", "মিরপুর ১০", "মতিঝিল", "গুলশান ২",
    "মোহাম্মদপুর", "ধানমন্ডি ১৫", "বাড্ডা", "খিলগাঁও",
    "যাত্রাবাড়ী", "কামরাঙ্গীরচর", "শ্যামলী", "লালবাগ",
    "পুরান ঢাকা", "আজিমপুর", "রামপুরা", "বনানী",
]

POLICE_STATIONS = [
    "উত্তরা পূর্ব থানা", "মিরপুর থানা", "মতিঝিল থানা", "গুলশান থানা",
    "মোহাম্মদপুর থানা", "ধানমন্ডি থানা", "বাড্ডা থানা", "খিলগাঁও থানা",
    "যাত্রাবাড়ী থানা", "লালবাগ থানা", "শ্যামলী থানা", "রামপুরা থানা",
]

TIMES = [
    "সকাল ৮টা", "সকাল ১০টা", "দুপুর ১২টা", "দুপুর ২টা",
    "বিকাল ৪টা", "বিকাল ৫টা", "সন্ধ্যা ৭টা", "রাত ৮টা",
    "রাত ১০টা", "রাত ১১টা", "গভীর রাত ১২টা", "ভোর ৫টা",
]

OFFENSE_TYPES = {
    "চুরি": {
        "section": "দণ্ডবিধি ধারা ৩৭৯",
        "description": "চুরি",
    },
    "ছিনতাই": {
        "section": "দণ্ডবিধি ধারা ৩৯২",
        "description": "ছিনতাই",
    },
    "মারধর": {
        "section": "দণ্ডবিধি ধারা ৩২৩",
        "description": "স্বেচ্ছাকৃত আঘাত",
    },
    "হুমকি": {
        "section": "দণ্ডবিধি ধারা ৫০৬",
        "description": "অপরাধমূলক ভয় দেখানো",
    },
    "প্রতারণা": {
        "section": "দণ্ডবিধি ধারা ৪২০",
        "description": "প্রতারণা",
    },
    "ধর্ষণ": {
        "section": "নারী ও শিশু নির্যাতন দমন আইন ২০০০ এর ধারা ৯",
        "description": "ধর্ষণ",
    },
    "অপহরণ": {
        "section": "দণ্ডবিধি ধারা ৩৬৩",
        "description": "অপহরণ",
    },
    "ডাকাতি": {
        "section": "দণ্ডবিধি ধারা ৩৯৫",
        "description": "ডাকাতি",
    },
}

STOLEN_ITEMS = [
    "মোবাইল ফোন", "মানিব্যাগ", "স্বর্ণের গহনা", "নগদ টাকা",
    "ল্যাপটপ", "ব্যাগ", "ঘড়ি", "সাইকেল", "মোটরসাইকেল",
]

AMOUNTS = [
    "৫,০০০ টাকা", "১০,০০০ টাকা", "১৫,০০০ টাকা", "২০,০০০ টাকা",
    "৫০,০০০ টাকা", "১,০০,০০০ টাকা", "২,০০০ টাকা", "৩০,০০০ টাকা",
]

DATES = []
base_date = datetime(2024, 1, 1)
for i in range(365):
    d = base_date + timedelta(days=i)
    DATES.append(d.strftime("%d/%m/%Y"))


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

def _random_date():
    return random.choice(DATES)


def _generate_theft_pair():
    victim = random.choice(VICTIM_NAMES)
    suspect = random.choice(SUSPECT_NAMES)
    location = random.choice(LOCATIONS)
    time = random.choice(TIMES)
    date = _random_date()
    item = random.choice(STOLEN_ITEMS)
    amount = random.choice(AMOUNTS)
    ps = random.choice(POLICE_STATIONS)
    offense = OFFENSE_TYPES["চুরি"]

    informal_variants = [
        f"আমার {item} চুরি হয়ে গেছে {location} তে {time} তে। {suspect} চুরি করেছে।",
        f"{date} তারিখে {time} এ {location} থেকে {suspect} আমার {item} চুরি করে নিয়ে গেছে।",
        f"{suspect} আমার {item} নিয়ে গেছে {location} থেকে। {time} এর দিকে ঘটনা হয়েছে।",
        f"আমি {victim}। {location} এলাকায় {time} তে আমার {item} চুরি হয়েছে। সন্দেহভাজন {suspect}।",
        f"গতকাল {time} তে {location} এ যাওয়ার সময় {suspect} আমার {item} চুরি করে পালিয়ে যায়।",
    ]

    informal = random.choice(informal_variants)

    formal = f"""বাংলাদেশ পুলিশ
প্রথম তথ্য বিবরণী (এফআইআর)

থানা: {ps}
এফআইআর নং: [স্বয়ংক্রিয়ভাবে বরাদ্দ]
তারিখ: {date}

অভিযোগকারীর তথ্য:
নাম: {victim}
ঘটনার তারিখ ও সময়: {date}, {time}
ঘটনাস্থল: {location}

অভিযুক্তের তথ্য:
নাম: {suspect}

অভিযোগের বিবরণ:
আমি নিম্নস্বাক্ষরকারী {victim} এতদ্বারা জানাচ্ছি যে, {date} তারিখ {time} সময়ে {location} এলাকায় অভিযুক্ত {suspect} আমার {item} (আনুমানিক মূল্য {amount}) চুরি করে পলায়ন করেছে।

প্রযোজ্য আইনি ধারা: {offense["section"]}
অপরাধের ধরন: {offense["description"]}

আমি উপরোক্ত ঘটনার সুষ্ঠু তদন্ত ও আইনানুগ ব্যবস্থা গ্রহণের জন্য আবেদন জানাচ্ছি।

অভিযোগকারীর স্বাক্ষর: {victim}
তারিখ: {date}"""

    return {"informal_text": informal, "formal_fir": formal, "offense_type": "চুরি", "date": date, "location": location}


def _generate_robbery_pair():
    victim = random.choice(VICTIM_NAMES)
    suspect = random.choice(SUSPECT_NAMES)
    location = random.choice(LOCATIONS)
    time = random.choice(TIMES)
    date = _random_date()
    item = random.choice(STOLEN_ITEMS)
    amount = random.choice(AMOUNTS)
    ps = random.choice(POLICE_STATIONS)
    offense = OFFENSE_TYPES["ছিনতাই"]

    informal_variants = [
        f"{location} এ {time} তে {suspect} আমার {item} ছিনিয়ে নিয়েছে। আমাকে ভয় দেখিয়েছে।",
        f"ছিনতাইকারী {suspect} {date} তারিখে {time} এ {location} এ আমার {item} কেড়ে নেয়।",
        f"রাস্তায় {location} এলাকায় {suspect} হঠাৎ এসে {time} তে আমার {item} ছিনিয়ে নেয়।",
        f"{date} তারিখ {time} সময়ে {location} এলাকায় {suspect} অস্ত্র দেখিয়ে আমার {item} লুট করে।",
    ]

    informal = random.choice(informal_variants)

    formal = f"""বাংলাদেশ পুলিশ
প্রথম তথ্য বিবরণী (এফআইআর)

থানা: {ps}
এফআইআর নং: [স্বয়ংক্রিয়ভাবে বরাদ্দ]
তারিখ: {date}

অভিযোগকারীর তথ্য:
নাম: {victim}
ঘটনার তারিখ ও সময়: {date}, {time}
ঘটনাস্থল: {location}

অভিযুক্তের তথ্য:
নাম: {suspect}

অভিযোগের বিবরণ:
আমি নিম্নস্বাক্ষরকারী {victim} এতদ্বারা জানাচ্ছি যে, {date} তারিখ {time} সময়ে {location} এলাকায় অভিযুক্ত {suspect} ভয় প্রদর্শন করে আমার {item} (আনুমানিক মূল্য {amount}) জোরপূর্বক ছিনিয়ে নিয়ে পলায়ন করেছে।

প্রযোজ্য আইনি ধারা: {offense["section"]}
অপরাধের ধরন: {offense["description"]}

আমি উপরোক্ত ঘটনার সুষ্ঠু তদন্ত ও আইনানুগ ব্যবস্থা গ্রহণের জন্য আবেদন জানাচ্ছি।

অভিযোগকারীর স্বাক্ষর: {victim}
তারিখ: {date}"""

    return {"informal_text": informal, "formal_fir": formal, "offense_type": "ছিনতাই", "date": date, "location": location}


def _generate_assault_pair():
    victim = random.choice(VICTIM_NAMES)
    suspect = random.choice(SUSPECT_NAMES)
    location = random.choice(LOCATIONS)
    time = random.choice(TIMES)
    date = _random_date()
    ps = random.choice(POLICE_STATIONS)
    offense = OFFENSE_TYPES["মারধর"]

    injury_types = ["মাথায় আঘাত", "হাতে আঘাত", "পেটে লাথি", "মুখে ঘুষি", "পিঠে আঘাত"]
    injury = random.choice(injury_types)

    informal_variants = [
        f"{suspect} আমাকে {location} এ {time} তে মেরেছে। {injury} পেয়েছি।",
        f"{date} তারিখে {time} এ {location} এলাকায় {suspect} আমাকে বেধড়ক মারধর করেছে।",
        f"{suspect} আমাকে হামলা করেছে {location} এ। {time} এ ঘটনা ঘটে। আমার {injury} হয়েছে।",
        f"{location} এলাকায় {time} তে {suspect} হঠাৎ আমার উপর ঝাঁপিয়ে পড়ে এবং {injury} করে।",
    ]

    informal = random.choice(informal_variants)

    formal = f"""বাংলাদেশ পুলিশ
প্রথম তথ্য বিবরণী (এফআইআর)

থানা: {ps}
এফআইআর নং: [স্বয়ংক্রিয়ভাবে বরাদ্দ]
তারিখ: {date}

অভিযোগকারীর তথ্য:
নাম: {victim}
ঘটনার তারিখ ও সময়: {date}, {time}
ঘটনাস্থল: {location}

অভিযুক্তের তথ্য:
নাম: {suspect}

অভিযোগের বিবরণ:
আমি নিম্নস্বাক্ষরকারী {victim} এতদ্বারা জানাচ্ছি যে, {date} তারিখ {time} সময়ে {location} এলাকায় অভিযুক্ত {suspect} বিনা উস্কানিতে আমার উপর শারীরিক আক্রমণ চালায় এবং {injury} ঘটায়। এতে আমি গুরুতর আহত হই।

প্রযোজ্য আইনি ধারা: {offense["section"]}
অপরাধের ধরন: {offense["description"]}

আমি উপরোক্ত ঘটনার সুষ্ঠু তদন্ত ও আইনানুগ ব্যবস্থা গ্রহণের জন্য আবেদন জানাচ্ছি।

অভিযোগকারীর স্বাক্ষর: {victim}
তারিখ: {date}"""

    return {"informal_text": informal, "formal_fir": formal, "offense_type": "মারধর", "date": date, "location": location}


def _generate_threat_pair():
    victim = random.choice(VICTIM_NAMES)
    suspect = random.choice(SUSPECT_NAMES)
    location = random.choice(LOCATIONS)
    time = random.choice(TIMES)
    date = _random_date()
    ps = random.choice(POLICE_STATIONS)
    offense = OFFENSE_TYPES["হুমকি"]

    threat_types = ["জীবননাশের হুমকি", "শারীরিক ক্ষতির হুমকি", "পরিবারকে ক্ষতি করার হুমকি", "বাড়ি ভাঙচুরের হুমকি"]
    threat = random.choice(threat_types)

    informal_variants = [
        f"{suspect} আমাকে {location} এ {time} তে {threat} দিয়েছে।",
        f"{date} তারিখে {suspect} ফোনে এবং সামনাসামনি আমাকে {threat} দিয়েছে।",
        f"{location} এলাকায় {time} তে {suspect} আমাকে সরাসরি {threat} দেয়।",
        f"{suspect} {date} তারিখে {time} এ আমার বাড়িতে এসে {threat} দিয়েছে।",
    ]

    informal = random.choice(informal_variants)

    formal = f"""বাংলাদেশ পুলিশ
প্রথম তথ্য বিবরণী (এফআইআর)

থানা: {ps}
এফআইআর নং: [স্বয়ংক্রিয়ভাবে বরাদ্দ]
তারিখ: {date}

অভিযোগকারীর তথ্য:
নাম: {victim}
ঘটনার তারিখ ও সময়: {date}, {time}
ঘটনাস্থল: {location}

অভিযুক্তের তথ্য:
নাম: {suspect}

অভিযোগের বিবরণ:
আমি নিম্নস্বাক্ষরকারী {victim} এতদ্বারা জানাচ্ছি যে, {date} তারিখ {time} সময়ে {location} এলাকায় অভিযুক্ত {suspect} আমাকে {threat} প্রদান করেছে। এতে আমি ও আমার পরিবার চরম আতঙ্কের মধ্যে রয়েছি।

প্রযোজ্য আইনি ধারা: {offense["section"]}
অপরাধের ধরন: {offense["description"]}

আমি উপরোক্ত ঘটনার সুষ্ঠু তদন্ত ও আইনানুগ ব্যবস্থা গ্রহণের জন্য আবেদন জানাচ্ছি।

অভিযোগকারীর স্বাক্ষর: {victim}
তারিখ: {date}"""

    return {"informal_text": informal, "formal_fir": formal, "offense_type": "হুমকি", "date": date, "location": location}


def _generate_fraud_pair():
    victim = random.choice(VICTIM_NAMES)
    suspect = random.choice(SUSPECT_NAMES)
    location = random.choice(LOCATIONS)
    time = random.choice(TIMES)
    date = _random_date()
    amount = random.choice(AMOUNTS)
    ps = random.choice(POLICE_STATIONS)
    offense = OFFENSE_TYPES["প্রতারণা"]

    fraud_types = [
        "চাকরি দেওয়ার প্রলোভনে",
        "জমি বিক্রির মিথ্যা প্রতিশ্রুতিতে",
        "ব্যবসায়িক অংশীদারিত্বের কথা বলে",
        "বিদেশ পাঠানোর প্রলোভনে",
        "বিনিয়োগে বেশি লাভের প্রতিশ্রুতিতে",
    ]
    fraud_type = random.choice(fraud_types)

    informal_variants = [
        f"{suspect} {fraud_type} আমার কাছ থেকে {amount} নিয়েছে কিন্তু কিছুই করেনি।",
        f"{date} তারিখে {suspect} {fraud_type} আমার {amount} প্রতারণামূলকভাবে নিয়ে যায়।",
        f"{location} এলাকার {suspect} {fraud_type} আমার {amount} নিয়েছে। এখন টাকা ফেরত দিচ্ছে না।",
        f"{suspect} আমার সাথে প্রতারণা করেছে। {fraud_type} {amount} নিয়েছে।",
    ]

    informal = random.choice(informal_variants)

    formal = f"""বাংলাদেশ পুলিশ
প্রথম তথ্য বিবরণী (এফআইআর)

থানা: {ps}
এফআইআর নং: [স্বয়ংক্রিয়ভাবে বরাদ্দ]
তারিখ: {date}

অভিযোগকারীর তথ্য:
নাম: {victim}
ঘটনার তারিখ ও সময়: {date}, {time}
ঘটনাস্থল: {location}

অভিযুক্তের তথ্য:
নাম: {suspect}

অভিযোগের বিবরণ:
আমি নিম্নস্বাক্ষরকারী {victim} এতদ্বারা জানাচ্ছি যে, অভিযুক্ত {suspect} {fraud_type} আমার নিকট হতে {amount} প্রতারণামূলকভাবে গ্রহণ করেছে এবং প্রতিশ্রুত কার্য সম্পাদনে ব্যর্থ হয়েছে। সর্বশেষ {date} তারিখে {time} সময়ে {location} এলাকায় টাকা ফেরত চাইলে অস্বীকার করে।

প্রযোজ্য আইনি ধারা: {offense["section"]}
অপরাধের ধরন: {offense["description"]}

আমি উপরোক্ত ঘটনার সুষ্ঠু তদন্ত ও আইনানুগ ব্যবস্থা গ্রহণের জন্য আবেদন জানাচ্ছি।

অভিযোগকারীর স্বাক্ষর: {victim}
তারিখ: {date}"""

    return {"informal_text": informal, "formal_fir": formal, "offense_type": "প্রতারণা", "date": date, "location": location}


def _generate_kidnapping_pair():
    victim = random.choice(VICTIM_NAMES)
    suspect = random.choice(SUSPECT_NAMES)
    location = random.choice(LOCATIONS)
    time = random.choice(TIMES)
    date = _random_date()
    ps = random.choice(POLICE_STATIONS)
    offense = OFFENSE_TYPES["অপহরণ"]

    victim_relation = random.choice(["আমার সন্তান", "আমার ভাই", "আমার বোন", "আমার স্ত্রী", "আমার স্বামী"])

    informal_variants = [
        f"{suspect} {date} তারিখে {time} এ {location} থেকে {victim_relation} কে অপহরণ করে নিয়ে গেছে।",
        f"{location} এলাকা থেকে {time} তে {suspect} {victim_relation} কে জোরপূর্বক নিয়ে গেছে।",
        f"{victim_relation} কে {suspect} অপহরণ করেছে {date} তারিখে {location} থেকে।",
    ]

    informal = random.choice(informal_variants)

    formal = f"""বাংলাদেশ পুলিশ
প্রথম তথ্য বিবরণী (এফআইআর)

থানা: {ps}
এফআইআর নং: [স্বয়ংক্রিয়ভাবে বরাদ্দ]
তারিখ: {date}

অভিযোগকারীর তথ্য:
নাম: {victim}
ঘটনার তারিখ ও সময়: {date}, {time}
ঘটনাস্থল: {location}

অভিযুক্তের তথ্য:
নাম: {suspect}

অভিযোগের বিবরণ:
আমি নিম্নস্বাক্ষরকারী {victim} এতদ্বারা জানাচ্ছি যে, {date} তারিখ {time} সময়ে {location} এলাকা থেকে অভিযুক্ত {suspect} {victim_relation} কে জোরপূর্বক অপহরণ করে নিয়ে যায়। অদ্যাবধি তার সন্ধান পাওয়া যায়নি।

প্রযোজ্য আইনি ধারা: {offense["section"]}
অপরাধের ধরন: {offense["description"]}

আমি উপরোক্ত ঘটনার সুষ্ঠু তদন্ত ও আইনানুগ ব্যবস্থা গ্রহণের জন্য আবেদন জানাচ্ছি।

অভিযোগকারীর স্বাক্ষর: {victim}
তারিখ: {date}"""

    return {"informal_text": informal, "formal_fir": formal, "offense_type": "অপহরণ", "date": date, "location": location}


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------

GENERATORS = [
    _generate_theft_pair,
    _generate_robbery_pair,
    _generate_assault_pair,
    _generate_threat_pair,
    _generate_fraud_pair,
    _generate_kidnapping_pair,
]


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> list[dict]:
    random.seed(seed)
    samples = []
    per_class = n_samples // len(GENERATORS)
    for gen in GENERATORS:
        for _ in range(per_class):
            samples.append(gen())
    # Fill remainder
    while len(samples) < n_samples:
        samples.append(random.choice(GENERATORS)())
    random.shuffle(samples)
    return samples


def build_instruction_pairs(samples: list[dict]) -> list[dict]:
    """Convert raw samples into instruction-tuning format."""
    instruction = (
        "নিচের অনানুষ্ঠানিক বাংলা বিবরণটি পড়ুন এবং একটি আনুষ্ঠানিক বাংলাদেশ পুলিশ "
        "এফআইআর (প্রথম তথ্য বিবরণী) তৈরি করুন। অভিযোগকারী, অভিযুক্ত, ঘটনার স্থান, "
        "সময়, তারিখ, এবং প্রযোজ্য আইনি ধারা স্পষ্টভাবে উল্লেখ করুন।"
    )
    pairs = []
    for s in samples:
        pairs.append(
            {
                "instruction": instruction,
                "input": s["informal_text"],
                "output": s["formal_fir"],
                "offense_type": s["offense_type"],
                "metadata": {
                    "date": s["date"],
                    "location": s["location"],
                },
            }
        )
    return pairs


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    samples = generate_dataset(n_samples=3000)
    pairs = build_instruction_pairs(samples)

    # Split train / val / test  80 / 10 / 10
    n = len(pairs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": pairs[:train_end],
        "validation": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split_name, data in splits.items():
        out_path = out_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples → {out_path}")

    print("\nSample record:")
    print(json.dumps(pairs[0], ensure_ascii=False, indent=2))
