"""Pydantic schema for structured legal complaint output."""

from pydantic import BaseModel


class LegalComplaint(BaseModel):
    incident_date: str
    incident_time: str
    location: str
    offense_type: str
    complainant_name: str
    accused_name: str
    summary_bn: str
    full_complaint_bn: str
    requested_action_bn: str
