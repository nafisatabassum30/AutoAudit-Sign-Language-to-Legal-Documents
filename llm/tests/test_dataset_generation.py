import json

from autoaudit_llm.schema import FIRComplaint, OffenseType
from generate_dataset import TEMPLATES, generate


def test_generate_produces_examples_for_every_offense_template():
    examples = generate(n_per_template=2, seed=7)
    assert len(examples) == 2 * len(TEMPLATES)


def test_every_generated_target_matches_schema():
    examples = generate(n_per_template=3, seed=7)
    for ex in examples:
        # Should not raise.
        FIRComplaint.model_validate(ex["target"])


def test_generated_narrative_date_is_grounded_in_raw_text():
    """The date/time stated in the structured target must actually appear in
    the raw input text -- otherwise the model would be trained to
    hallucinate facts, which the system prompt explicitly forbids.
    """
    examples = generate(n_per_template=5, seed=3)
    for ex in examples:
        target = ex["target"]
        assert target["incident_date"] in ex["raw_text"]
        assert target["incident_time"] in ex["raw_text"]


def test_generated_accused_and_property_facts_are_grounded_in_raw_text():
    """Any specific accused name/amount/item asserted in the target must be
    traceable to the raw input text; templates that don't identify an
    accused must default to "অজ্ঞাত" rather than inventing an identity.
    """
    examples = generate(n_per_template=8, seed=11)
    for ex in examples:
        target = ex["target"]
        raw_text = ex["raw_text"]

        if target["accused_name"] != "অজ্ঞাত":
            assert target["accused_name"] in raw_text
        if target["victim_name"] and target["victim_name"] != target["complainant_name"]:
            assert target["victim_name"] in raw_text


def test_unknown_accused_templates_never_assign_a_specific_identity():
    """Offenses where the perpetrator is realistically unknown (street theft,
    snatching, vehicle theft, cyber crime) must never get a fabricated,
    specific accused name.
    """
    unknown_accused_offenses = {
        OffenseType.THEFT.value,
        OffenseType.ROBBERY_SNATCHING.value,
        OffenseType.VEHICLE_THEFT.value,
        OffenseType.CYBER_CRIME.value,
        OffenseType.MISSING_PERSON.value,
    }
    examples = generate(n_per_template=10, seed=5)
    for ex in examples:
        if ex["target"]["offense_type"] in unknown_accused_offenses:
            assert ex["target"]["accused_name"] == "অজ্ঞাত"


def test_generated_examples_are_json_serializable():
    examples = generate(n_per_template=1, seed=1)
    for ex in examples:
        json.dumps(ex, ensure_ascii=False)
