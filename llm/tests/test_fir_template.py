from src.fir_template import render_fir_document
from src.schema import FIRRecord, OffenseType, Person, PropertyItem


def test_render_includes_key_fields():
    record = FIRRecord(
        offense_type=OffenseType.THEFT,
        complainant=Person(name="করিম উদ্দিন", address="উত্তরা, ঢাকা"),
        accused_unknown=True,
        incident_date="2026-06-15",
        incident_time="বিকাল ৫টা",
        incident_location="উত্তরা",
        police_station="উত্তরা পশ্চিম থানা",
        district="ঢাকা",
        stolen_or_damaged_items=[PropertyItem(description="মানিব্যাগ", estimated_value_bdt=1500)],
        narrative_bn="অভিযোগকারীর মানিব্যাগ চুরি হয়েছে।",
    )
    doc = render_fir_document(record, filing_date="2026-06-16")

    assert "করিম উদ্দিন" in doc
    assert "উত্তরা পশ্চিম থানা" in doc
    assert "মানিব্যাগ" in doc
    assert "1,500" in doc
    assert "চুরি" in doc
    assert "2026-06-16" in doc


def test_render_handles_missing_optional_fields():
    record = FIRRecord(
        offense_type=OffenseType.MISSING_PERSON,
        complainant=Person(name="রহিমা বেগম"),
        incident_location="মিরপুর",
        narrative_bn="একজন ব্যক্তি নিখোঁজ।",
    )
    doc = render_fir_document(record)
    assert "মিরপুর" in doc
    assert "উল্লেখ নেই" in doc
