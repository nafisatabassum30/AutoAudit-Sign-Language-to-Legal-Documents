import json

from evaluate import evaluate
from src.data_generation import generate_examples


def test_evaluate_perfect_model_scores_100_percent():
    rows = generate_examples(20, seed=11)

    def perfect_generate_fn(input_text: str) -> str:
        row = next(r for r in rows if r["input_text"] == input_text)
        return json.dumps(row["output_json"], ensure_ascii=False)

    report = evaluate(perfect_generate_fn, rows)
    metrics = report["metrics"]
    assert metrics["json_parse_success_rate"] == 1.0
    for field, acc in metrics["field_accuracy"].items():
        assert acc == 1.0, f"field {field} expected perfect accuracy, got {acc}"
    assert metrics["avg_narrative_token_f1"] == 1.0


def test_evaluate_broken_model_reports_failures():
    rows = generate_examples(5, seed=12)

    def broken_generate_fn(input_text: str) -> str:
        return "এটি কোনো বৈধ JSON নয়"

    report = evaluate(broken_generate_fn, rows)
    assert report["metrics"]["json_parse_success_rate"] == 0.0
    assert all("error" in ex for ex in report["per_example"])
