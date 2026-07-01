#!/usr/bin/env python3
"""QLoRA fine-tuning of a Bangla-capable base LLM for FIR generation.

Converts recognized (informal) Bangla text from the BdSL ST-GNN stage into
a structured, schema-validated FIR JSON record.

Example
-------
    python train.py --config configs/training_config.yaml

Requires a CUDA GPU with bitsandbytes support. See ``llm/README.md`` for
setup and hardware notes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--lora-config", default="configs/lora_config.yaml")
    parser.add_argument(
        "--base-model", default=None, help="Override base_model from the training config"
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    lora_cfg = load_config(args.lora_config)
    if args.base_model:
        cfg["base_model"] = args.base_model

    # Heavy ML deps are imported lazily so that `--help`, config validation,
    # and the rest of this repo's tooling work without a GPU/torch installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from src.dataset import load_hf_dataset

    base_model = cfg["base_model"]
    print(f"Loading base model: {base_model}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_dir = Path(cfg["data_dir"])
    train_ds = load_hf_dataset(data_dir, "train")
    val_ds = load_hf_dataset(data_dir, "val")
    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    def formatting_func(example) -> str:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 50),
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 3),
        seed=cfg.get("seed", 42),
        max_seq_length=cfg.get("max_seq_length", 1024),
        packing=cfg.get("packing", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        bf16=cfg.get("bf16", True),
        report_to=cfg.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Saved LoRA adapter + tokenizer to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
