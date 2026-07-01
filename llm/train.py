# -*- coding: utf-8 -*-
"""Fine-tune a causal LM with (Q)LoRA to turn raw Bangla BdSL-derived text
into a structured, FIR-ready JSON legal complaint.

Usage:
    python train.py --config configs/lora_qlora.yaml

    # Quick CPU smoke test with a tiny model + tiny dataset (no GPU needed):
    python train.py --config configs/smoke_test.yaml

Any field in the YAML config can be overridden from the CLI, e.g.:
    python train.py --config configs/lora_qlora.yaml --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --num_train_epochs 1
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from autoaudit_llm.dataset import FIRInstructionDataset, PadCollator

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    # Common overrides, kept explicit for a friendlier --help; anything else
    # in the YAML can still be edited directly.
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Override total steps (useful for smoke tests).")
    return parser


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    for key in ["base_model", "output_dir", "train_file", "val_file", "num_train_epochs", "per_device_train_batch_size"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def build_model_and_tokenizer(config: dict):
    base_model = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    quant_config = None
    if config.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=DTYPE_MAP[config.get("bnb_4bit_compute_dtype", "bfloat16")],
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        )

    model_kwargs = {}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if config.get("use_lora", True):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("lora_target_modules"),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main() -> None:
    args = build_arg_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)

    os.makedirs(config["output_dir"], exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(config)

    max_seq_length = config.get("max_seq_length", 1024)
    train_dataset = FIRInstructionDataset(config["train_file"], tokenizer, max_seq_length)
    eval_dataset = None
    if config.get("val_file") and os.path.exists(config["val_file"]):
        eval_dataset = FIRInstructionDataset(config["val_file"], tokenizer, max_seq_length)

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        num_train_epochs=config.get("num_train_epochs", 3),
        max_steps=args.max_steps if args.max_steps is not None else -1,
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging_steps", 10),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.get("eval_steps", 50),
        save_strategy="steps",
        save_steps=config.get("save_steps", 50),
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=config.get("seed", 42),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()

    final_dir = os.path.join(config["output_dir"], "final")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved adapter + tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
