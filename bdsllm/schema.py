from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IncidentFacts:
    """Normalized facts passed from sign recognition into the LLM layer."""

    recognized_text: str
    complainant_name: str = ""
    complainant_address: str = ""
    complainant_phone: str = ""
    incident_date: str = ""
    incident_time: str = ""
    incident_location: str = ""
    accused_name: str = ""
    accused_details: str = ""
    offense_type: str = ""
    requested_action: str = "আইনগত ব্যবস্থা গ্রহণের অনুরোধ করছি।"
    additional_context: str = ""

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "IncidentFacts":
        return cls(
            recognized_text=str(item.get("recognized_text") or item.get("sign_text") or "").strip(),
            complainant_name=str(item.get("complainant_name", "")).strip(),
            complainant_address=str(item.get("complainant_address", "")).strip(),
            complainant_phone=str(item.get("complainant_phone", "")).strip(),
            incident_date=str(item.get("incident_date", "")).strip(),
            incident_time=str(item.get("incident_time", "")).strip(),
            incident_location=str(item.get("incident_location", "")).strip(),
            accused_name=str(item.get("accused_name", "")).strip(),
            accused_details=str(item.get("accused_details", "")).strip(),
            offense_type=str(item.get("offense_type", "")).strip(),
            requested_action=str(
                item.get("requested_action", "আইনগত ব্যবস্থা গ্রহণের অনুরোধ করছি।")
            ).strip(),
            additional_context=str(item.get("additional_context", "")).strip(),
        )

    def to_prompt_fields(self) -> dict[str, str]:
        return {
            "recognized_text": self.recognized_text,
            "complainant_name": self.complainant_name or "অজ্ঞাত/প্রদান করা হয়নি",
            "complainant_address": self.complainant_address or "প্রদান করা হয়নি",
            "complainant_phone": self.complainant_phone or "প্রদান করা হয়নি",
            "incident_date": self.incident_date or "প্রদান করা হয়নি",
            "incident_time": self.incident_time or "প্রদান করা হয়নি",
            "incident_location": self.incident_location or "প্রদান করা হয়নি",
            "accused_name": self.accused_name or "অজ্ঞাত",
            "accused_details": self.accused_details or "প্রদান করা হয়নি",
            "offense_type": self.offense_type or "প্রাথমিকভাবে নির্ধারিত নয়",
            "requested_action": self.requested_action,
            "additional_context": self.additional_context or "নেই",
        }


@dataclass(slots=True)
class ComplaintExample:
    """Instruction-tuning example in chat/instruction format."""

    instruction: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instruction": self.instruction,
            "response": self.response,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload
