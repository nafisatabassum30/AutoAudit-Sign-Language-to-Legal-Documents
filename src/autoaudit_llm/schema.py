"""Structured schemas for the FIR (First Information Report) generation stage.

Two models are defined:

* :class:`FIRExtraction` — the entities the LLM must identify from the raw,
  informal Bangla text produced by the sign-language recognition stage
  (victim, suspect, location, time, offense type, ...).
* :class:`FIRComplaint`  — the final FIR-ready legal document. It contains all
  fields required by a Bangladeshi FIR plus a formatted Bangla narrative.

Using :mod:`pydantic` gives us validation, JSON (de)serialization and a stable
contract shared by the training data format, the inference engine and the API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FIRExtraction(BaseModel):
    """Entities extracted from the raw signed statement.

    All fields are optional because the source (sign-language) statement is
    often terse and may omit details. ``None``/empty means "not stated".
    """

    complainant_name: Optional[str] = Field(
        default=None, description="বাদী / অভিযোগকারীর নাম (complainant name)"
    )
    victim_name: Optional[str] = Field(
        default=None, description="ভুক্তভোগীর নাম (victim name)"
    )
    accused_name: Optional[str] = Field(
        default=None, description="অভিযুক্ত ব্যক্তির নাম/পরিচয় (accused/suspect)"
    )
    offense_type: Optional[str] = Field(
        default=None, description="অপরাধের ধরন, যেমন চুরি, ছিনতাই, হামলা (offense type)"
    )
    location: Optional[str] = Field(
        default=None, description="ঘটনাস্থল (place of occurrence)"
    )
    incident_date: Optional[str] = Field(
        default=None, description="ঘটনার তারিখ (date of incident)"
    )
    incident_time: Optional[str] = Field(
        default=None, description="ঘটনার সময় (time of occurrence)"
    )
    stolen_items: List[str] = Field(
        default_factory=list,
        description="ক্ষতিগ্রস্ত/চুরি হওয়া জিনিসপত্র (items stolen/damaged)",
    )
    description: Optional[str] = Field(
        default=None, description="ঘটনার সংক্ষিপ্ত বিবরণ (short raw description)"
    )


class FIRComplaint(BaseModel):
    """A complete, FIR-ready legal complaint in Bangla."""

    offense_type: str = Field(description="অপরাধের ধরন")
    incident_date: str = Field(description="ঘটনার তারিখ")
    incident_time: str = Field(description="ঘটনার সময়")
    location: str = Field(description="ঘটনার স্থান / ঠিকানা")
    complainant_name: str = Field(description="বাদীর নাম")
    victim_name: str = Field(description="ভুক্তভোগীর নাম")
    accused_name: str = Field(description="অভিযুক্তের নাম/পরিচয়")
    stolen_items: List[str] = Field(default_factory=list)
    complaint_body: str = Field(
        description="আনুষ্ঠানিক আইনি ভাষায় অভিযোগের পূর্ণ বিবরণ (formal narrative)"
    )

    def to_document(self) -> str:
        """Render the complaint as a human-readable, FIR-styled Bangla document."""
        items = "、".join(self.stolen_items) if self.stolen_items else "প্রযোজ্য নয়"
        return (
            "প্রথম তথ্য বিবরণী (এফআইআর)\n"
            "==============================\n"
            f"অপরাধের ধরন   : {self.offense_type}\n"
            f"ঘটনার তারিখ   : {self.incident_date}\n"
            f"ঘটনার সময়     : {self.incident_time}\n"
            f"ঘটনাস্থল      : {self.location}\n"
            f"বাদী          : {self.complainant_name}\n"
            f"ভুক্তভোগী      : {self.victim_name}\n"
            f"অভিযুক্ত       : {self.accused_name}\n"
            f"ক্ষতিগ্রস্ত সামগ্রী : {items}\n"
            "------------------------------\n"
            "বিবরণ:\n"
            f"{self.complaint_body}\n"
        )
