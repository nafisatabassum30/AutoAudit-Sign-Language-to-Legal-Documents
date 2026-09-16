import argparse
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Bengali-capable seq2seq model for FIR generation")
    parser.add_argument("--train_file", type=Path, required=True, help="Path to JSONL training data")
    parser.add_argument("--output_dir", type=Path, default=Path("output/llm"), help="Directory to save the fine-tuned model")
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-small", help="Base model name or path")
    parser.add_argument("--max_source_length", type=int, default=512, help="Maximum token length for the source prompt")
    parser.add_argument("--max_target_length", type=int, default=384, help="Maximum token length for the FIR output")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optionally limit training samples for quick experiments")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_instruction_prompt(example):
    instruction = example.get("instruction", "একটি পূর্ণাঙ্গ বাংলা FIR/অভিযোগপত্র তৈরি করুন")
    input_text = example.get("input", "")
    prompt = (
        f"নির্দেশ: {instruction}. শুধু FIR/অভিযোগপত্রের বিষয়বস্তু লিখুন, কোনো অতিরিক্ত ব্যাখ্যা দেবেন না.\n"
        f"ঘটনার বিবরণ: {input_text}\n"
        "FIR/অভিযোগপত্র:"
    )
    output_text = example.get("output", "")
    return prompt, output_text


def preprocess(dataset, tokenizer, max_source_length, max_target_length):
    def _tokenize(example):
        prompt, output_text = build_instruction_prompt(example)
        model_inputs = tokenizer(prompt, truncation=True, max_length=max_source_length, padding=False)
        labels = tokenizer(output_text, truncation=True, max_length=max_target_length, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_tokenize, batched=False, remove_columns=dataset.column_names)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    model_name = args.model_name_or_path
    fallback_model = "google/flan-t5-small"

    try:
        print(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as exc:
        print(f"Failed to load requested model {model_name}: {exc}")
        print(f"Falling back to {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model, use_fast=False)
        model_name = fallback_model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("CUDA is not available; using CPU-friendly full-precision loading")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    print(f"Loading training data from {args.train_file}")
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    tokenized_dataset = preprocess(dataset, tokenizer, args.max_source_length, args.max_target_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting training")
    trainer.train()
    print(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
