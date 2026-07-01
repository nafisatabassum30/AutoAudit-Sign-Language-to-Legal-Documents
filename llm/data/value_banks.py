"""Value banks (name pools, locations, time/date phrases) used by the
synthetic FIR dataset generator.

These are plain Python data structures with no external dependencies so the
generator can run fully offline.
"""

from __future__ import annotations

COMPLAINANT_NAMES_MALE = [
    "মোঃ রফিকুল ইসলাম",
    "মোঃ কামরুল হাসান",
    "আব্দুল করিম",
    "মোঃ জাহাঙ্গীর আলম",
    "মোঃ শাহাদাত হোসেন",
    "রাকিব হাসান",
    "তানভীর আহমেদ",
    "মোঃ নাসির উদ্দিন",
    "ফারুক হোসেন",
    "মোঃ সোহেল রানা",
    "ইমরান হোসেন",
    "আরিফুল ইসলাম",
    "মোঃ হাবিবুর রহমান",
    "মোঃ আলমগীর হোসেন",
    "সাইফুল ইসলাম",
]

COMPLAINANT_NAMES_FEMALE = [
    "ফাতেমা বেগম",
    "সালমা আক্তার",
    "রোকসানা পারভীন",
    "নাসরিন আক্তার",
    "শিরিন সুলতানা",
    "মোছাঃ রাবেয়া খাতুন",
    "তাসলিমা বেগম",
    "সুমাইয়া ইসলাম",
    "রহিমা খাতুন",
    "জেসমিন আক্তার",
    "মোছাঃ আয়েশা সিদ্দিকা",
    "নাজমা বেগম",
    "ফরিদা ইয়াসমিন",
    "শাহনাজ পারভীন",
    "মিনারা বেগম",
]

ALL_NAMES = COMPLAINANT_NAMES_MALE + COMPLAINANT_NAMES_FEMALE

# Dhaka Metropolitan Police thana / area names (also usable as place-of-occurrence).
THANAS = [
    "উত্তরা",
    "মিরপুর",
    "ধানমন্ডি",
    "গুলশান",
    "মোহাম্মদপুর",
    "বাড্ডা",
    "তেজগাঁও",
    "রমনা",
    "শাহবাগ",
    "খিলগাঁও",
    "যাত্রাবাড়ী",
    "ওয়ারী",
    "লালবাগ",
    "পল্টন",
    "কোতোয়ালী",
    "ডেমরা",
    "সবুজবাগ",
    "বনানী",
    "ভাটারা",
    "উত্তরখান",
]

DISTRICTS = [
    "ঢাকা",
    "গাজীপুর",
    "নারায়ণগঞ্জ",
    "চট্টগ্রাম",
    "রাজশাহী",
    "খুলনা",
    "সিলেট",
    "বরিশাল",
    "রংপুর",
    "ময়মনসিংহ",
]

# time_phrase -> 24h "HH:MM" used to build a resolvable time_of_occurrence.
TIME_PHRASES = {
    "সকাল ৮টা": "08:00",
    "সকাল ৯টা": "09:00",
    "সকাল ১০টা": "10:00",
    "দুপুর ১২টা": "12:00",
    "দুপুর ১টা": "13:00",
    "দুপুর ২টা": "14:00",
    "বিকাল ৩টা": "15:00",
    "বিকাল ৪টা": "16:00",
    "বিকাল ৫টা": "17:00",
    "সন্ধ্যা ৬টা": "18:00",
    "সন্ধ্যা ৭টা": "19:00",
    "রাত ৮টা": "20:00",
    "রাত ৯টা": "21:00",
    "রাত ১০টা": "22:00",
    "রাত ১১টা": "23:00",
    "রাত ১২টা": "00:00",
    "ভোর ৫টা": "05:00",
}

# date_phrase -> day offset relative to the report date (0 = today, negative = past).
# "পরশু" is treated here as "the day before yesterday" (a common past-tense usage).
DATE_PHRASES_FIXED_OFFSET = {
    "আজ": 0,
    "গতকাল": -1,
    "গত রাতে": -1,
    "পরশু": -2,
    "তিন দিন আগে": -3,
    "এক সপ্তাহ আগে": -7,
}

# "গত <বার>" phrases resolve to the most recent past occurrence of that weekday.
WEEKDAY_PHRASES = {
    "গত রবিবার": 6,
    "গত সোমবার": 0,
    "গত মঙ্গলবার": 1,
    "গত বুধবার": 2,
    "গত বৃহস্পতিবার": 3,
    "গত শুক্রবার": 4,
    "গত শনিবার": 5,
}

ALL_DATE_PHRASES = list(DATE_PHRASES_FIXED_OFFSET) + list(WEEKDAY_PHRASES)

STOLEN_ITEMS = [
    "মোবাইল ফোন",
    "মানিব্যাগ",
    "নগদ টাকা",
    "ল্যাপটপ",
    "স্বর্ণের চেইন",
    "হাতঘড়ি",
    "কাঁধের ব্যাগ",
    "সাইকেল",
    "মোটরসাইকেল",
    "নগদ ৫০,০০০ টাকা",
    "স্বর্ণের কানের দুল",
    "এটিএম কার্ড ও নগদ টাকা",
]

DAMAGED_PROPERTY_ITEMS = [
    "দোকানের সাটার",
    "গাড়ির কাঁচ",
    "বসতঘরের জানালা",
    "সীমানা প্রাচীর",
    "দোকানের মালামাল",
    "মোটরসাইকেল",
]

RELATION_PHRASES_KNOWN_SUSPECT = [
    "প্রতিবেশী",
    "পূর্ব পরিচিত ব্যক্তি",
    "সহকর্মী",
    "সাবেক ব্যবসায়িক অংশীদার",
    "একই এলাকার বাসিন্দা",
]
