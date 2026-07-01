import argparse
import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Bangla LLM with QLoRA for legal complaints")
    parser.add_argument("--train_file", type=Path, required=True, help="Path to JSONL training data")
    parser.add_argument("--output_dir", type=Path, default=Path("output/llm"), help="Directory to save the fine-tuned model")
    parser.add_argument("--model_name_or_path", type=str, default="sagorsarker/bangla-llama-3b", help="Base model name or path")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum token length for each example")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optionally limit training samples for quick experiments")
    parser.add_argument("--dataset_format", type=str, choices=["auto", "prompt_completion", "text_only"], default="auto", help="Training dataset format")
    return parser.parse_args()


def build_instruction_prompt(example):
    instruction = example.get("instruction", "একটি বাংলা আইনি অভিযোগ / FIR তৈরি করুন")
    input_text = example.get("input", "")
    if input_text:
        prompt = (
            f"নির্দেশ: {instruction}\n"
            f"আইনগত ঘটনার সারসংক্ষেপ: {input_text}\n"
            "FIR:"
        )
    else:
        prompt = f"নির্দেশ: {instruction}\nFIR:"

    output_text = example.get("output", "")
    return prompt, output_text


def detect_dataset_format(example):
    if "text" in example:
        return "text_only"
    if "instruction" in example and "input" in example and "output" in example:
        return "prompt_completion"
    return "text_only"


def tokenize_example(example, tokenizer, max_length, dataset_format):
    if dataset_format == "text_only":
        text = example.get("text", "").strip()
        full_text = f"{text} {tokenizer.eos_token or ''}".strip()
        tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"],
        }

    prompt, output_text = build_instruction_prompt(example)
    full_text = f"{prompt} {output_text} {tokenizer.eos_token or ''}".strip()
    tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)

    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()
    prompt_length = len(tokenized_prompt["input_ids"])
    labels[:prompt_length] = [-100] * prompt_length

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }


def preprocess(dataset, tokenizer, max_length, dataset_format):
    if dataset_format == "auto":
        first = dataset[0]
        dataset_format = detect_dataset_format(first)
        print(f"Auto-detected dataset format: {dataset_format}")

    def _tokenize(example):
        return tokenize_example(example, tokenizer, max_length, dataset_format)

    tokenized = dataset.map(
        _tokenize,
        batched=False,
        remove_columns=dataset.column_names,
    )
    return tokenized


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with 4-bit quantization support")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)

    print(f"Loading training data from {args.train_file}")
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    tokenized_dataset = preprocess(dataset, tokenizer, args.max_length, args.dataset_format)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        evaluation_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training")
    trainer.train()
    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
