"""Catalog of FIR offense types used by the synthetic dataset generator.

Each entry describes one type of complaint the deaf/BdSL user might sign about.
It provides:
  - ``raw_templates``: informal, telegraphic Bangla sentence templates that
    approximate what the upstream ST-GNN sign-recognition stage would output
    (e.g. "আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা").
  - ``narrative_template``: a formal Bangla legal narrative paragraph template
    that the fine-tuned LLM should learn to produce for the FIR document.

NOTE: The penal-code sections listed here are illustrative examples commonly
associated with each offense category in Bangladesh and are meant to help the
model learn realistic FIR structure. They are NOT a substitute for legal
review -- any production deployment must have final documents checked by a
qualified legal professional before submission to law enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OffenseType:
    key: str
    name_bn: str
    sections: list[str]
    raw_templates: list[str]
    narrative_template: str
    needs_item: bool = False
    item_bank: str | None = None  # "stolen" | "damaged" | None
    is_missing_person: bool = False


OFFENSE_TYPES: list[OffenseType] = [
    OffenseType(
        key="theft",
        name_bn="চুরি",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৩৭৯"],
        needs_item=True,
        item_bank="stolen",
        raw_templates=[
            "আমার {item} চুরি {location} {time}",
            "{item} খোয়া {location} {date} {time}",
            "কেউ {item} চুরি করেছে {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} আনুমানিক {time} সময়ে {location} "
            "এলাকায় {suspect_phrase} সংগোপনে তার {item} চুরি করে নিয়ে যায়। ঘটনাটি "
            "জানার সাথে সাথে আশেপাশে খোঁজাখুঁজি করা হলেও অভিযুক্তের সন্ধান পাওয়া যায়নি। "
            "এই ঘটনায় অভিযোগকারী আর্থিকভাবে ক্ষতিগ্রস্ত হয়েছেন এবং উপযুক্ত ব্যবস্থা "
            "গ্রহণের জন্য থানায় অভিযোগ দায়ের করছেন।"
        ),
    ),
    OffenseType(
        key="house_burglary",
        name_bn="গৃহে চুরি",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৩৮০", "দণ্ডবিধি ১৮৬০ - ধারা ৪৫৭"],
        needs_item=True,
        item_bank="stolen",
        raw_templates=[
            "বাড়িতে চুরি {location} রাতে {item}",
            "ঘরের তালা ভেঙে {item} নিয়ে গেছে {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} রাতে আনুমানিক {time} সময়ে তার "
            "বসতবাড়ির তালা ভেঙে {suspect_phrase} ঘরে প্রবেশ করে {item} চুরি করে নিয়ে "
            "যায়। ঘটনাটি {location} এলাকায় সংঘটিত হয়। এতে অভিযোগকারী মারাত্মকভাবে "
            "আর্থিক ক্ষতির শিকার হয়েছেন এবং আইনি প্রতিকারের জন্য থানায় অভিযোগ দায়ের "
            "করছেন।"
        ),
    ),
    OffenseType(
        key="robbery_snatching",
        name_bn="ছিনতাই",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৩৯২"],
        needs_item=True,
        item_bank="stolen",
        raw_templates=[
            "রাস্তায় ছিনতাই {item} {location} {time}",
            "মোটরসাইকেলে এসে {item} ছিনিয়ে নিয়েছে {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} আনুমানিক {time} সময়ে {location} "
            "এলাকায় রাস্তা দিয়ে যাওয়ার সময় {suspect_phrase} জোরপূর্বক ভয়ভীতি প্রদর্শন "
            "করে তার {item} ছিনিয়ে নিয়ে যায়। ঘটনার প্রতিবাদ করায় অভিযোগকারীকে ধাক্কা "
            "দিয়ে পালিয়ে যায় অভিযুক্তরা। এই ঘটনায় ন্যায়বিচার প্রার্থনা করে থানায় অভিযোগ "
            "দায়ের করা হলো।"
        ),
    ),
    OffenseType(
        key="dacoity",
        name_bn="ডাকাতি",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৩৯৫"],
        needs_item=True,
        item_bank="stolen",
        raw_templates=[
            "দলবেঁধে ডাকাতি ঘরে {location} রাতে",
            "অস্ত্র নিয়ে ডাকাতি বাড়িতে {item} {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} গভীর রাতে আনুমানিক {time} সময়ে "
            "{location} এলাকায় তার বাড়িতে {suspect_phrase} দলবদ্ধভাবে অস্ত্রের ভয় দেখিয়ে "
            "প্রবেশ করে এবং জোরপূর্বক {item} লুট করে নিয়ে যায়। এই ঘটনায় পরিবারের "
            "সদস্যরা চরম আতঙ্কের মধ্যে পড়েন। উপযুক্ত আইনি ব্যবস্থা গ্রহণের জন্য থানায় "
            "অভিযোগ দায়ের করা হলো।"
        ),
    ),
    OffenseType(
        key="assault",
        name_bn="মারামারি ও শারীরিক আঘাত",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৩২৩", "দণ্ডবিধি ১৮৬০ - ধারা ৩২৫"],
        raw_templates=[
            "মারামারি হয়েছে আমার সাথে {location} {time}",
            "কেউ মেরেছে আমাকে {location} {date}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} আনুমানিক {time} সময়ে {location} "
            "এলাকায় {suspect_phrase} তুচ্ছ ঘটনার জেরে অভিযোগকারীর উপর অতর্কিতে হামলা "
            "চালিয়ে শারীরিকভাবে আঘাত করে। এতে অভিযোগকারী শারীরিকভাবে জখম হন এবং "
            "স্থানীয় হাসপাতালে চিকিৎসা গ্রহণ করেন। এই ঘটনায় আইনি ব্যবস্থা গ্রহণের জন্য "
            "থানায় অভিযোগ দায়ের করা হলো।"
        ),
    ),
    OffenseType(
        key="threat_intimidation",
        name_bn="হুমকি ও ভয়ভীতি প্রদর্শন",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৫০৬"],
        raw_templates=[
            "হুমকি দিয়েছে মারার {location}",
            "প্রাণনাশের হুমকি {date} {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} {location} এলাকায় {suspect_phrase} "
            "অভিযোগকারীকে প্রাণনাশের হুমকি ও ভয়ভীতি প্রদর্শন করে। এই ঘটনায় অভিযোগকারী "
            "চরম নিরাপত্তাহীনতায় ভুগছেন এবং নিজের ও পরিবারের জীবনের সুরক্ষার জন্য থানায় "
            "অভিযোগ দায়ের করছেন।"
        ),
    ),
    OffenseType(
        key="fraud",
        name_bn="প্রতারণা ও জালিয়াতি",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৪২০", "দণ্ডবিধি ১৮৬০ - ধারা ৪০৬"],
        needs_item=True,
        item_bank="stolen",
        raw_templates=[
            "টাকা প্রতারণা {item} {location}",
            "কেউ প্রতারণা করে টাকা নিয়েছে {location} {date}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} {location} এলাকায় {suspect_phrase} "
            "মিথ্যা প্রতিশ্রুতি দিয়ে অভিযোগকারীর নিকট থেকে {item} প্রতারণামূলকভাবে গ্রহণ "
            "করে এবং পরবর্তীতে তা ফেরত দেয়নি বা কোনো প্রতিশ্রুতি পালন করেনি। এতে "
            "অভিযোগকারী মারাত্মক আর্থিক ক্ষতির শিকার হয়েছেন এবং প্রতারকের বিরুদ্ধে "
            "আইনগত ব্যবস্থা গ্রহণের জন্য থানায় অভিযোগ দায়ের করছেন।"
        ),
    ),
    OffenseType(
        key="sexual_harassment",
        name_bn="যৌন হয়রানি",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৩৫৪", "দণ্ডবিধি ১৮৬০ - ধারা ৫০৯"],
        raw_templates=[
            "যৌন হয়রানি করেছে {location} {time}",
            "উৎপীড়ন করেছে আমাকে রাস্তায় {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} আনুমানিক {time} সময়ে {location} "
            "এলাকায় {suspect_phrase} অভিযোগকারীর সাথে অশোভন আচরণ করে, যা তার "
            "শ্লীলতাহানির শামিল। এই ঘটনায় অভিযোগকারী মানসিকভাবে গুরুতরভাবে "
            "আঘাতপ্রাপ্ত হয়েছেন এবং অভিযুক্তের বিরুদ্ধে আইনগত ব্যবস্থা গ্রহণের জন্য থানায় "
            "অভিযোগ দায়ের করছেন।"
        ),
    ),
    OffenseType(
        key="eve_teasing",
        name_bn="ইভটিজিং",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৫০৯"],
        raw_templates=[
            "রাস্তায় উত্যক্ত করে {location} স্কুলের সামনে",
            "ইভটিজিং করে {location} প্রতিদিন",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, দীর্ঘদিন ধরে {location} এলাকায় "
            "{suspect_phrase} যাতায়াতের পথে অভিযোগকারীকে উত্যক্ত করে আসছে এবং "
            "কুরুচিপূর্ণ মন্তব্য করছে। {date} এই আচরণ চরম পর্যায়ে পৌঁছালে অভিযোগকারী "
            "আতঙ্কিত হয়ে পড়েন। এই ঘটনায় প্রতিকারের জন্য থানায় অভিযোগ দায়ের করা হলো।"
        ),
    ),
    OffenseType(
        key="missing_person",
        name_bn="নিখোঁজ ব্যক্তি",
        sections=["সাধারণ ডায়েরি (জিডি)"],
        is_missing_person=True,
        raw_templates=[
            "আমার ভাই নিখোঁজ {date} {location}",
            "{victim} বাড়ি ফেরেনি {date}",
            "খুঁজে পাচ্ছি না {victim} {location} {date}",
        ],
        narrative_template=(
            "অভিযোগকারী জানান যে, তার নিকটাত্মীয় {victim} {date} আনুমানিক {time} "
            "সময়ে {location} এলাকা থেকে বের হওয়ার পর থেকে নিখোঁজ রয়েছেন। পরিবারের "
            "সদস্যরা আত্মীয়স্বজন, পরিচিতজন ও সম্ভাব্য স্থানে খোঁজাখুঁজি করেও তার কোনো "
            "সন্ধান পাননি। নিখোঁজ ব্যক্তির সন্ধান ও নিরাপত্তার বিষয়ে প্রয়োজনীয় ব্যবস্থা "
            "গ্রহণের জন্য থানায় সাধারণ ডায়েরি (জিডি) দায়ের করা হলো।"
        ),
    ),
    OffenseType(
        key="property_damage",
        name_bn="সম্পত্তি ভাংচুর ও ক্ষতিসাধন",
        sections=["দণ্ডবিধি ১৮৬০ - ধারা ৪২৭"],
        needs_item=True,
        item_bank="damaged",
        raw_templates=[
            "ভাংচুর করেছে দোকান {location}",
            "কেউ ভেঙে ফেলেছে {item} {location} {date}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} আনুমানিক {time} সময়ে {location} "
            "এলাকায় {suspect_phrase} উদ্দেশ্যপ্রণোদিতভাবে অভিযোগকারীর {item} ভাংচুর "
            "করে ক্ষতিসাধন করে। এই ঘটনায় অভিযোগকারী আর্থিকভাবে ক্ষতিগ্রস্ত হয়েছেন এবং "
            "ক্ষতিপূরণ ও আইনি ব্যবস্থা গ্রহণের জন্য থানায় অভিযোগ দায়ের করছেন।"
        ),
    ),
    OffenseType(
        key="drug_related",
        name_bn="মাদক সংক্রান্ত অভিযোগ",
        sections=["মাদকদ্রব্য নিয়ন্ত্রণ আইন, ২০১৮ - ধারা ৩৬"],
        raw_templates=[
            "মাদক বিক্রি হয় {location} এলাকায়",
            "মাদক ব্যবসা করে চক্র {location}",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {location} এলাকায় {suspect_phrase} "
            "প্রকাশ্যে মাদকদ্রব্য বিক্রি ও সরবরাহ করে আসছে, যা এলাকার আইনশৃঙ্খলা "
            "পরিস্থিতি ও তরুণ সমাজের উপর ক্ষতিকর প্রভাব ফেলছে। এই বিষয়ে উপযুক্ত "
            "ব্যবস্থা গ্রহণের জন্য থানায় অভিযোগ দায়ের করা হলো।"
        ),
    ),
    OffenseType(
        key="cyber_harassment",
        name_bn="সাইবার হয়রানি",
        sections=["সাইবার সুরক্ষা আইন, ২০২৩ - ধারা ২৫"],
        raw_templates=[
            "ফেসবুকে ছবি বিকৃত করে পোস্ট করেছে",
            "অনলাইনে হয়রানি করছে মেসেজ পাঠিয়ে",
        ],
        narrative_template=(
            "অভিযোগকারী {victim} জানান যে, {date} থেকে {suspect_phrase} সামাজিক "
            "যোগাযোগমাধ্যমে অভিযোগকারীর ব্যক্তিগত ছবি ও তথ্য বিকৃতভাবে প্রকাশ করে এবং "
            "ক্ষতিকর বার্তা প্রেরণ করে হয়রানি করে আসছে। এতে অভিযোগকারী মানসিক ক্ষতির "
            "শিকার হয়েছেন এবং আইনগত প্রতিকারের জন্য থানায় অভিযোগ দায়ের করছেন।"
        ),
    ),
]

OFFENSE_BY_KEY: dict[str, OffenseType] = {o.key: o for o in OFFENSE_TYPES}
