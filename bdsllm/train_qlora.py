from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


@dataclass(slots=True)
class TrainArgs:
    model_name: str
    train_file: str
    output_dir: str
    eval_file: str | None = None
    max_seq_length: int = 2048
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8


def _format_example(example: dict[str, str], tokenizer: AutoTokenizer) -> str:
    instruction = example["instruction"].strip()
    response = example["response"].strip()
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{instruction}\n\nউত্তর:\n{response}{tokenizer.eos_token or ''}"


def train(args: TrainArgs) -> None:
    data_files: dict[str, str] = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)

    def tokenize(example: dict[str, str]) -> dict[str, list[int]]:
        text = _format_example(example, tokenizer)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)
    eval_strategy = "epoch" if args.eval_file else "no"

    trainer = Trainer(
        model=model,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy=eval_strategy,
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Fine-tune a Bangla legal complaint adapter with QLoRA.")
    parser.add_argument("--model-name", required=True, help="Base causal LM, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train-file", required=True, help="Instruction JSONL from scripts/prepare_llm_dataset.py")
    parser.add_argument("--output-dir", required=True, help="Directory for the LoRA adapter")
    parser.add_argument("--eval-file", help="Optional validation JSONL")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parsed = parser.parse_args()
    return TrainArgs(**vars(parsed))


if __name__ == "__main__":
    train(parse_args())
