# -*- coding: utf-8 -*-
"""Slot vocabularies used to synthesize BdSL-derived Bangla text -> FIR
training pairs.

These lists intentionally mirror the kind of short, telegraphic phrases the
upstream ST-GNN sign-recognition stage is expected to emit (concatenated
recognized sign glosses), e.g. "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫টা", rather than
grammatically complete sentences. The fine-tuned LLM's job is to turn that
into properly structured, formal legal Bangla.

NOTE: This is a *synthetic bootstrap* dataset used to get the LLM training
pipeline working end-to-end before real collected FIR templates (see the
project README's "Collect 1,000+ FIR templates" step) are available. Once
real templates are collected (e.g. via BLAST / local police stations), drop
them into ``llm/data/raw_fir_templates/`` and extend ``generate_dataset.py``
or write a small converter to merge them in -- the training/inference code
does not need to change.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

COMPLAINANT_NAMES = [
    "মোঃ রফিকুল ইসলাম",
    "সালমা আক্তার",
    "জাহাঙ্গীর আলম",
    "রোকেয়া বেগম",
    "ইমরান হোসেন",
    "তানভীর আহমেদ",
    "নাসরিন সুলতানা",
    "আব্দুল করিম",
    "ফাতেমা খাতুন",
    "শারমিন আক্তার",
    "জসিম উদ্দিন",
    "আয়েশা সিদ্দিকা",
    "মোঃ সোহরাব হোসেন",
    "লাভলী বেগম",
    "কামরুল হাসান",
]

VICTIM_NAMES = COMPLAINANT_NAMES + [
    "তানিয়া ইসলাম (৭ বছর)",
    "রাশেদুল করিম (১০ বছর)",
    "মোঃ আলাউদ্দিন (৬৫ বছর)",
]

ACCUSED_NAMES = [
    "প্রতিবেশী মোঃ শাহজাহান",
    "সহকর্মী রাশেদ",
    "পরিচিত ব্যক্তি সোহেল",
    "প্রতিবেশী আলমগীর",
    "মোঃ সাইফুল ইসলাম",
]

HUSBAND_NAMES = [
    "স্বামী মোঃ হাবিবুর রহমান",
    "স্বামী মোঃ কামাল হোসেন",
    "স্বামী মোঃ আব্দুল্লাহ আল মামুন",
]

LOCATIONS = [
    "উত্তরা, ঢাকা",
    "মিরপুর-১০, ঢাকা",
    "ধানমন্ডি, ঢাকা",
    "গুলশান-২, ঢাকা",
    "মোহাম্মদপুর, ঢাকা",
    "যাত্রাবাড়ী, ঢাকা",
    "বাড্ডা, ঢাকা",
    "সাভার, ঢাকা",
    "নারায়ণগঞ্জ সদর",
    "চকবাজার, চট্টগ্রাম",
    "আগ্রাবাদ, চট্টগ্রাম",
    "জিন্দাবাজার, সিলেট",
    "বোয়ালিয়া, রাজশাহী",
    "খালিশপুর, খুলনা",
    "কোতোয়ালি, কুমিল্লা",
    "টাঙ্গাইল সদর",
    "গাজীপুর সদর",
]

THANA_BY_LOCATION = {
    "উত্তরা, ঢাকা": "উত্তরা পশ্চিম থানা",
    "মিরপুর-১০, ঢাকা": "মিরপুর থানা",
    "ধানমন্ডি, ঢাকা": "ধানমন্ডি থানা",
    "গুলশান-২, ঢাকা": "গুলশান থানা",
    "মোহাম্মদপুর, ঢাকা": "মোহাম্মদপুর থানা",
    "যাত্রাবাড়ী, ঢাকা": "যাত্রাবাড়ী থানা",
    "বাড্ডা, ঢাকা": "বাড্ডা থানা",
    "সাভার, ঢাকা": "সাভার থানা",
    "নারায়ণগঞ্জ সদর": "নারায়ণগঞ্জ সদর থানা",
    "চকবাজার, চট্টগ্রাম": "চকবাজার থানা",
    "আগ্রাবাদ, চট্টগ্রাম": "পাঁচলাইশ থানা",
    "জিন্দাবাজার, সিলেট": "কোতোয়ালি থানা, সিলেট",
    "বোয়ালিয়া, রাজশাহী": "বোয়ালিয়া থানা",
    "খালিশপুর, খুলনা": "খালিশপুর থানা",
    "কোতোয়ালি, কুমিল্লা": "কোতোয়ালি মডেল থানা, কুমিল্লা",
    "টাঙ্গাইল সদর": "টাঙ্গাইল সদর থানা",
    "গাজীপুর সদর": "গাজীপুর সদর থানা",
}

TIMES = [
    "সকাল ৮টা",
    "সকাল ৯টা ৩০ মিনিট",
    "দুপুর ১২টা",
    "দুপুর ২টা",
    "বিকেল ৪টা",
    "বিকেল ৫টা",
    "সন্ধ্যা ৭টা",
    "রাত ৯টা",
    "রাত ১১টা",
    "ভোর ৫টা",
]

ITEMS = [
    "ওয়ালেট",
    "মোবাইল ফোন",
    "ল্যাপটপ",
    "মোটরসাইকেল",
    "সাইকেল",
    "কাঁধের ব্যাগ",
    "নগদ ৫০,০০০ টাকা",
    "স্বর্ণের গলার হার",
    "রিকশা",
    "সিএনজি অটোরিকশা",
]

AMOUNTS = ["১০,০০০ টাকা", "৫০,০০০ টাকা", "১,০০,০০০ টাকা", "২,৫০,০০০ টাকা", "৩০,০০০ টাকা"]

WITNESS_NAME_POOL = [
    "প্রতিবেশী মোঃ কালাম",
    "দোকানদার আব্দুল হাই",
    "বোন সুমি আক্তার",
    "রিকশাচালক মোঃ ইদ্রিস",
]


def random_recent_date(rng: random.Random, max_days_ago: int = 120) -> str:
    delta = rng.randint(0, max_days_ago)
    d = date.today() - timedelta(days=delta)
    return d.strftime("%d %B %Y")


def maybe(rng: random.Random, value, p: float = 0.7):
    """Return ``value`` with probability ``p``, else ``None`` (field omitted)."""
    return value if rng.random() < p else None
