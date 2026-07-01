"""
Data preprocessing pipeline for the BdSL → Bangla FIR LLM.

Responsibilities:
  - Load JSON instruction-tuning data
  - Tokenise with the target model's tokenizer
  - Format into prompt templates (Alpaca / ChatML)
  - Produce HuggingFace Dataset objects ready for training
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

ALPACA_PROMPT = """নিচের নির্দেশনাটি একটি কাজ বর্ণনা করে। অনুরোধটি যথাযথভাবে সম্পন্ন করে একটি উত্তর লিখুন।

### নির্দেশনা:
{instruction}

### ইনপুট:
{input}

### আউটপুট:
{output}"""

ALPACA_PROMPT_INFERENCE = """নিচের নির্দেশনাটি একটি কাজ বর্ণনা করে। অনুরোধটি যথাযথভাবে সম্পন্ন করে একটি উত্তর লিখুন।

### নির্দেশনা:
{instruction}

### ইনপুট:
{input}

### আউটপুট:
"""

DEFAULT_INSTRUCTION = (
    "নিচের অনানুষ্ঠানিক বাংলা বিবরণটি পড়ুন এবং একটি আনুষ্ঠানিক বাংলাদেশ পুলিশ "
    "এফআইআর (প্রথম তথ্য বিবরণী) তৈরি করুন। অভিযোগকারী, অভিযুক্ত, ঘটনার স্থান, "
    "সময়, তারিখ, এবং প্রযোজ্য আইনি ধারা স্পষ্টভাবে উল্লেখ করুন।"
)


def format_prompt(instruction: str, input_text: str, output_text: str = "") -> str:
    if output_text:
        return ALPACA_PROMPT.format(instruction=instruction, input=input_text, output=output_text)
    return ALPACA_PROMPT_INFERENCE.format(instruction=instruction, input=input_text)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_json_split(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_dict(data_dir: str | Path) -> DatasetDict:
    data_dir = Path(data_dir)
    splits = {}
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.json"
        if path.exists():
            records = load_json_split(path)
            splits[split] = Dataset.from_list(records)
            logger.info("Loaded %d records for split '%s'", len(records), split)
        else:
            logger.warning("Split file not found: %s", path)
    return DatasetDict(splits)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

class FIRDatasetProcessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 1024,
        padding: str = "max_length",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize_example(self, example: dict) -> dict:
        instruction = example.get("instruction", DEFAULT_INSTRUCTION)
        input_text = example["input"]
        output_text = example.get("output", "")

        full_prompt = format_prompt(instruction, input_text, output_text)

        tokenized = self.tokenizer(
            full_prompt,
            max_length=self.max_seq_length,
            truncation=True,
            padding=self.padding,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Build labels: mask prompt tokens, keep output tokens only
        if output_text:
            prompt_only = format_prompt(instruction, input_text, "")
            prompt_len = len(
                self.tokenizer(
                    prompt_only,
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors=None,
                )["input_ids"]
            )
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[: self.max_seq_length]
            if len(labels) < self.max_seq_length:
                labels += [-100] * (self.max_seq_length - len(labels))
        else:
            labels = input_ids[:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def process_dataset(self, dataset: Dataset, num_proc: int = 4) -> Dataset:
        return dataset.map(
            self._tokenize_example,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="Tokenising",
        )

    def process_dataset_dict(self, dataset_dict: DatasetDict, num_proc: int = 4) -> DatasetDict:
        return DatasetDict(
            {
                split: self.process_dataset(ds, num_proc=num_proc)
                for split, ds in dataset_dict.items()
            }
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Preprocess FIR dataset")
    parser.add_argument("--data_dir", default="data/synthetic", help="Directory with train/val/test JSON files")
    parser.add_argument("--model_name", default="unsloth/llama-3-8b-bnb-4bit", help="Tokenizer name/path")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    dataset_dict = load_dataset_dict(args.data_dir)

    processor = FIRDatasetProcessor(tokenizer, max_seq_length=args.max_seq_length)
    processed = processor.process_dataset_dict(dataset_dict)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    logger.info("Saved processed dataset to %s", out_dir)
