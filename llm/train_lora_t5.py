import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a T5/mT5 model for Bangla legal complaint generation"
    )

    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("output/llm_lora_v1"))

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/mt5-small",
        help="Base model. google/mt5-small is multilingual and suitable for Bangla.",
    )

    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=384)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(example):
    instruction = str(
        example.get("instruction", "বাংলা আইনগত অভিযোগপত্র তৈরি করুন")
    ).strip()

    input_text = str(example.get("input", "")).strip()

    prompt = (
        f"নির্দেশ: {instruction}\n"
        f"ইনপুট: {input_text}\n"
        "উত্তর:"
    )

    target = str(example.get("output", "")).strip()

    return prompt, target


def preprocess_function(example, tokenizer, max_source_length, max_target_length):
    prompt, target = build_prompt(example)

    model_inputs = tokenizer(
        prompt,
        max_length=max_source_length,
        truncation=True,
        padding=False,
    )

    labels = tokenizer(
        target,
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def load_base_model(model_name_or_path):
    # Safe training mode: use float32.
    # This avoids fp16 NaN loss / NaN grad_norm problem on mT5.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    )

    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading tokenizer:", args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model:", args.model_name_or_path)

    model = load_base_model(args.model_name_or_path)

    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)

    print("\nLoRA trainable parameters:")
    model.print_trainable_parameters()

    print("\nLoading dataset:", args.train_file)

    dataset = load_dataset(
        "json",
        data_files=str(args.train_file),
        split="train",
    )

    if args.max_train_samples is not None:
        max_count = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(max_count))

    print("Dataset size:", len(dataset))

    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(
            ex,
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        ),
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,

        # Important safe settings
        fp16=False,
        max_grad_norm=1.0,
        optim="adamw_torch",

        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStarting LoRA fine-tuning...")

    trainer.train()

    print("\nSaving LoRA adapter to:", args.output_dir)

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    info = {
        "base_model": args.model_name_or_path,
        "train_file": str(args.train_file),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ["q", "v"],
        "task_type": "SEQ_2_SEQ_LM",
        "fp16": False,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        "learning_rate": args.learning_rate,
    }

    with open(args.output_dir / "lora_training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Check these files:")
    print(args.output_dir / "adapter_config.json")
    print(args.output_dir / "adapter_model.safetensors")
    print(args.output_dir / "tokenizer_config.json")


if __name__ == "__main__":
    main()