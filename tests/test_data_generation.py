from autoaudit_llm.data_generation import generate_dataset, generate_example
from autoaudit_llm.schema import FIRComplaint, FIRExtraction
import random


def test_generate_dataset_is_deterministic():
    a = generate_dataset(50, seed=7)
    b = generate_dataset(50, seed=7)
    assert a == b


def test_generate_dataset_size_and_uniqueness():
    ds = generate_dataset(100, seed=1)
    assert len(ds) == 100
    statements = [ex["raw_statement"] for ex in ds]
    assert len(set(statements)) == len(statements)


def test_example_shape_validates_against_schema():
    ex = generate_example(random.Random(3))
    assert set(ex) == {"raw_statement", "extraction", "complaint"}
    # both structures must validate
    FIRExtraction(**ex["extraction"])
    complaint = FIRComplaint(**ex["complaint"])
    assert complaint.offense_type
    assert complaint.complaint_body
