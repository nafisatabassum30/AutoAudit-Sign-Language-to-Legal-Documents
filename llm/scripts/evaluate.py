"""
Evaluation script for the fine-tuned BdSL FIR LLM.

Metrics computed:
  - BLEU-4 (sacrebleu)
  - ROUGE-1/2/L (rouge-score)
  - Entity F1 per field (complainant, accused, location, date, time, legal_section)
  - FIR field coverage (fraction of required fields present)

Usage:
    python scripts/evaluate.py \
        --model_path models/checkpoints/final_adapter \
        --data_path  data/synthetic/test.json \
        --output     results/eval_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import FIRGenerator, InferenceConfig, extract_entities
from src.utils import compute_bleu, compute_rouge, compute_entity_f1, setup_logging

logger = logging.getLogger(__name__)

REQUIRED_FIR_FIELDS = [
    "থানা",
    "অভিযোগকারীর তথ্য",
    "অভিযুক্তের তথ্য",
    "ঘটনার তারিখ",
    "ঘটনাস্থল",
    "প্রযোজ্য আইনি ধারা",
    "অভিযোগের বিবরণ",
]

ENTITY_FIELDS = ["complainant", "accused", "location", "date", "time", "legal_section"]


def field_coverage(fir_text: str) -> float:
    """Fraction of required FIR fields present in generated text."""
    present = sum(1 for f in REQUIRED_FIR_FIELDS if f in fir_text)
    return present / len(REQUIRED_FIR_FIELDS)


def evaluate(
    generator: FIRGenerator,
    test_data: list[dict],
    max_samples: int = 200,
) -> dict:
    test_data = test_data[:max_samples]
    n = len(test_data)
    logger.info("Evaluating on %d samples...", n)

    predictions, references = [], []
    pred_entities_list, gold_entities_list = [], []
    coverage_scores = []

    for i, sample in enumerate(test_data):
        if i % 20 == 0:
            logger.info("  Progress: %d/%d", i, n)

        pred_fir = generator.generate(sample["input"])
        ref_fir = sample["output"]

        predictions.append(pred_fir)
        references.append(ref_fir)

        pred_ents = extract_entities(pred_fir)
        gold_ents = extract_entities(ref_fir)
        pred_entities_list.append(pred_ents.__dict__)
        gold_entities_list.append(gold_ents.__dict__)

        coverage_scores.append(field_coverage(pred_fir))

    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)
    entity_f1 = compute_entity_f1(pred_entities_list, gold_entities_list, ENTITY_FIELDS)
    avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0

    results = {
        "n_samples": n,
        "bleu": round(bleu, 4),
        "rouge": rouge,
        "entity_f1": entity_f1,
        "avg_field_coverage": round(avg_coverage, 4),
    }
    return results


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate BdSL FIR LLM")
    parser.add_argument("--model_path", default="models/checkpoints/final_adapter")
    parser.add_argument("--base_model", default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument("--data_path", default="data/synthetic/test.json")
    parser.add_argument("--output", default="results/eval_results.json")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    args = parser.parse_args()

    # Load test data
    with open(args.data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info("Loaded %d test samples from %s", len(test_data), args.data_path)

    # Build generator
    cfg = InferenceConfig(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )
    generator = FIRGenerator(cfg)

    results = evaluate(generator, test_data, max_samples=args.max_samples)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Results ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
