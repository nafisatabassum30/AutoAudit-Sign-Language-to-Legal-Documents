from llm.infer_llm import build_fir_document


def test_build_fir_document_contains_expected_sections():
    complaint = build_fir_document("আমার ওয়ালেট বিকেলে চুরি হয়ে গেছে।")

    assert "ফৌজদারি অভিযোগ / FIR" in complaint
    assert "ঘটনার বিবরণ:" in complaint
    assert "অভিযুক্ত ব্যক্তি বা প্রতিষ্ঠান:" in complaint
    assert "বাদী এই ঘটনার বিষয়ে" in complaint
    assert "চুরি হয়ে গেছে" in complaint
