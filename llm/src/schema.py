"""Structured data contract for FIR (First Information Report) generation.

The fine-tuned LLM is trained to read informal, sign-language-derived Bangla
text (e.g. ``"আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়"``) and emit a JSON
object that conforms to :class:`FIRRecord`. Keeping a strict schema lets us:

* Validate every model generation (catch hallucinated/missing fields).
* Repair minor JSON mistakes deterministically (see ``src/postprocess.py``).
* Render the same record into a human-readable Bangla legal document
  (see ``src/fir_template.py``) independent of how the LLM phrased things.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class OffenseType(str, Enum):
    THEFT = "চুরি"
    ROBBERY = "ডাকাতি/ছিনতাই"
    ASSAULT = "মারামারি/আঘাত"
    FRAUD = "প্রতারণা"
    MISSING_PERSON = "নিখোঁজ ব্যক্তি"
    HARASSMENT = "যৌন হয়রানি/উত্তক্ত করা"
    DOMESTIC_VIOLENCE = "গৃহ নির্যাতন"
    PROPERTY_DAMAGE = "সম্পত্তি ক্ষতি"
    THREAT = "হুমকি"
    OTHER = "অন্যান্য"


# Rough, non-exhaustive mapping from offense type to commonly cited
# Bangladesh Penal Code sections. This is only used as a *default suggestion*
# in generated documents -- it must always be reviewed by a human officer /
# legal professional before filing.
DEFAULT_PENAL_CODE_SECTIONS = {
    OffenseType.THEFT: ["দণ্ডবিধি ধারা ৩৭৯ (চুরি)"],
    OffenseType.ROBBERY: ["দণ্ডবিধি ধারা ৩৯২ (ডাকাতি)", "দণ্ডবিধি ধারা ৩৯৭"],
    OffenseType.ASSAULT: ["দণ্ডবিধি ধারা ৩২৩ (আঘাত)", "দণ্ডবিধি ধারা ৩২৫"],
    OffenseType.FRAUD: ["দণ্ডবিধি ধারা ৪২০ (প্রতারণা)"],
    OffenseType.MISSING_PERSON: ["ফৌজদারি কার্যবিধি ধারা ১৫৪ (সাধারণ ডায়েরি)"],
    OffenseType.HARASSMENT: ["দণ্ডবিধি ধারা ৩৫৪এ", "নারী ও শিশু নির্যাতন দমন আইন"],
    OffenseType.DOMESTIC_VIOLENCE: ["পারিবারিক সহিংসতা (প্রতিরোধ ও সুরক্ষা) আইন, ২০১০"],
    OffenseType.PROPERTY_DAMAGE: ["দণ্ডবিধি ধারা ৪২৭ (ক্ষতিসাধন)"],
    OffenseType.THREAT: ["দণ্ডবিধি ধারা ৫০৬ (ভীতি প্রদর্শন)"],
    OffenseType.OTHER: [],
}


class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="ব্যক্তির নাম")
    address: Optional[str] = Field(default=None, description="ঠিকানা")
    phone: Optional[str] = Field(default=None, description="মোবাইল নম্বর")
    nid: Optional[str] = Field(default=None, description="জাতীয় পরিচয়পত্র নম্বর")
    relation_to_victim: Optional[str] = Field(
        default=None, description="ভুক্তভোগীর সাথে সম্পর্ক (যদি অভিযোগকারী আলাদা হয়)"
    )


class PropertyItem(BaseModel):
    description: str = Field(description="সম্পত্তি/জিনিসের বিবরণ, যেমন 'মানিব্যাগ'")
    estimated_value_bdt: Optional[float] = Field(
        default=None, description="আনুমানিক মূল্য (টাকায়)"
    )
    quantity: Optional[int] = Field(default=1)


class FIRRecord(BaseModel):
    """Canonical, schema-validated representation of one FIR."""

    offense_type: OffenseType
    penal_code_sections: List[str] = Field(default_factory=list)

    complainant: Person
    victim: Optional[Person] = Field(
        default=None, description="ভুক্তভোগী (অভিযোগকারী থেকে ভিন্ন হলে)"
    )
    accused: Optional[Person] = Field(
        default=None, description="অভিযুক্ত ব্যক্তি (অজ্ঞাত হলে None)"
    )
    accused_unknown: bool = Field(
        default=False, description="অভিযুক্ত অজ্ঞাত/অচিহ্নিত কিনা"
    )

    incident_date: Optional[str] = Field(
        default=None, description="ঘটনার তারিখ, ফরম্যাট YYYY-MM-DD (আনুমানিক হলেও গ্রহণযোগ্য)"
    )
    incident_time: Optional[str] = Field(
        default=None, description="ঘটনার সময়, ফরম্যাট HH:MM বা বর্ণনামূলক (যেমন 'বিকাল ৫টা')"
    )
    incident_location: str = Field(description="ঘটনাস্থল")
    police_station: Optional[str] = Field(default=None, description="সংশ্লিষ্ট থানা")
    district: Optional[str] = Field(default=None, description="জেলা")

    stolen_or_damaged_items: List[PropertyItem] = Field(default_factory=list)
    witnesses: List[str] = Field(default_factory=list)

    narrative_bn: str = Field(
        description="ঘটনার আনুষ্ঠানিক ও আইনি ভাষায় লিখিত বিবরণ (অনুচ্ছেদ)"
    )
    raw_input_text: Optional[str] = Field(
        default=None, description="স্বাক্ষর ভাষা থেকে প্রাপ্ত মূল/অপরিমার্জিত বাংলা টেক্সট"
    )

    @model_validator(mode="after")
    def _default_sections(self) -> "FIRRecord":
        if not self.penal_code_sections:
            self.penal_code_sections = list(
                DEFAULT_PENAL_CODE_SECTIONS.get(self.offense_type, [])
            )
        return self

    def with_defaults_filled(self) -> "FIRRecord":
        """Return a copy with sensible defaults applied (e.g. victim==complainant)."""
        data = self.model_dump()
        if data.get("victim") is None:
            data["victim"] = data["complainant"]
        if not data.get("penal_code_sections"):
            data["penal_code_sections"] = list(
                DEFAULT_PENAL_CODE_SECTIONS.get(self.offense_type, [])
            )
        return FIRRecord.model_validate(data)
