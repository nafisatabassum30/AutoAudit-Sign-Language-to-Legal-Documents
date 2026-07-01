# -*- coding: utf-8 -*-
"""Structured schema for the FIR (First Information Report) legal complaint.

This is the contract between the fine-tuned LLM and the rest of the
AutoAudit pipeline. The LLM is trained to always emit a single JSON object
matching this schema. Keeping the *keys* in English (while the *values* are
Bangla) makes the output trivial to parse/validate deterministically and
keeps the model's generation budget focused on the Bangla content instead of
re-learning key names in two languages.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class OffenseType(str, Enum):
    """Offense categories recognized by the pipeline.

    Modeled after the categories most commonly seen in Bangladesh Police
    FIR / GD templates and BLAST complaint drafting guides.
    """

    THEFT = "চুরি"
    ROBBERY_SNATCHING = "ছিনতাই"
    DACOITY = "ডাকাতি"
    FRAUD = "প্রতারণা"
    ASSAULT = "মারধর/শারীরিক নির্যাতন"
    SEXUAL_HARASSMENT = "যৌন নির্যাতন"
    KIDNAPPING = "অপহরণ"
    MISSING_PERSON = "নিখোঁজ/হারানো ব্যক্তি"
    VEHICLE_THEFT = "যানবাহন চুরি"
    CYBER_CRIME = "সাইবার অপরাধ"
    THREAT = "হুমকি"
    DOWRY_HARASSMENT = "যৌতুকের জন্য নির্যাতন"
    OTHER = "অন্যান্য"


class FIRComplaint(BaseModel):
    """Structured representation of one FIR-ready legal complaint."""

    complainant_name: str = Field(
        default="অজ্ঞাত", description="অভিযোগকারীর নাম"
    )
    complainant_address: Optional[str] = Field(
        default=None, description="অভিযোগকারীর ঠিকানা"
    )
    complainant_phone: Optional[str] = Field(
        default=None, description="অভিযোগকারীর মোবাইল নম্বর"
    )
    victim_name: Optional[str] = Field(
        default=None,
        description="ভিকটিমের নাম (অভিযোগকারী থেকে ভিন্ন হলে, যেমন নিখোঁজ সংবাদে)",
    )
    incident_date: Optional[str] = Field(
        default=None, description="ঘটনার তারিখ (YYYY-MM-DD অথবা বাংলা তারিখ)"
    )
    incident_time: Optional[str] = Field(default=None, description="ঘটনার সময়")
    incident_location: str = Field(
        default="উল্লেখ নেই", description="ঘটনাস্থল / এলাকা"
    )
    thana: Optional[str] = Field(
        default=None, description="সংশ্লিষ্ট থানার নাম (অনুমান করা হলে)"
    )
    offense_type: OffenseType = Field(description="অপরাধের ধরন")
    accused_name: str = Field(
        default="অজ্ঞাত", description="অভিযুক্তের নাম, না জানা থাকলে 'অজ্ঞাত'"
    )
    accused_description: Optional[str] = Field(
        default=None, description="অভিযুক্তের শারীরিক বর্ণনা / সনাক্তকরণ তথ্য"
    )
    property_description: Optional[str] = Field(
        default=None, description="খোয়া যাওয়া/ক্ষতিগ্রস্ত সম্পত্তির বিবরণ"
    )
    witnesses: List[str] = Field(
        default_factory=list, description="সাক্ষীদের নাম (থাকলে)"
    )
    narrative: str = Field(
        description="ঘটনার সম্পূর্ণ, আনুষ্ঠানিক আইনি ভাষায় লিখিত বর্ণনা (বাংলায়)"
    )

    @field_validator("offense_type", mode="before")
    @classmethod
    def _coerce_offense_type(cls, value: object) -> object:
        if isinstance(value, OffenseType):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            for member in OffenseType:
                if normalized in (member.value, member.name):
                    return member
        return value

    model_config = {"use_enum_values": True}


REQUIRED_KEYS = [
    "complainant_name",
    "incident_location",
    "offense_type",
    "accused_name",
    "narrative",
]

ALL_KEYS = list(FIRComplaint.model_fields.keys())
