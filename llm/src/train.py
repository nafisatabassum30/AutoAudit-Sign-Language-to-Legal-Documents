#!/usr/bin/env python3
"""QLoRA / LoRA fine-tuning entry point for the BdSL-raw-text -> FIR-JSON LLM.

Example:
    python -m src.train --config config/training_config.yaml

    # Quick CPU smoke test with a tiny model and a handful of steps:
    python -m src.train --config config/training_config.yaml \\
        --base-model sshleifer/tiny-gpt2 --no-4bit --max-steps 5 \\
        --lora-target-modules c_attn --output-dir /tmp/fir-lora-smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FIRDataCollator, build_sft_dataset  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--base-model", default=None, help="Override base_model from config")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Cap training steps (useful for smoke tests)")
    parser.add_argument("--per-device-batch-size", type=int, default=None, help="Override training.per_device_train_batch_size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Override training.gradient_accumulation_steps")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (e.g. on CPU)")
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated override for lora.target_modules (needed for non-Llama/Qwen architectures)",
    )
    return parser.parse_args()


def maybe_build_quant_config(cfg: dict, disable: bool):
    if disable or not cfg.get("load_in_4bit", False) or not torch.cuda.is_available():
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    base_model = args.base_model or cfg["base_model"]
    output_dir = args.output_dir or cfg["output_dir"]
    train_path = args.train_path or cfg["data"]["train_path"]
    val_path = args.val_path or cfg["data"]["val_path"]
    max_seq_length = cfg["data"].get("max_seq_length", 1024)

    print(f"[train] base_model={base_model} output_dir={output_dir}")

    quant_cfg = maybe_build_quant_config(cfg.get("quantization", {}), disable=args.no_4bit)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs = {"trust_remote_code": True}
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules")
    if args.lora_target_modules:
        target_modules = args.lora_target_modules.split(",")

    if target_modules:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if quant_cfg is not None:
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("[train] No LoRA target modules resolved; doing full fine-tuning.")

    train_ds = build_sft_dataset(train_path, tokenizer, max_seq_length)
    eval_ds = build_sft_dataset(val_path, tokenizer, max_seq_length) if os.path.exists(val_path) else None

    train_cfg = cfg.get("training", {})
    per_device_batch_size = args.per_device_batch_size or train_cfg.get("per_device_train_batch_size", 4)
    grad_accum_steps = args.gradient_accumulation_steps or train_cfg.get("gradient_accumulation_steps", 4)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "epoch") if eval_ds is not None else "no",
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        bf16=train_cfg.get("bf16", False) and torch.cuda.is_available(),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
        max_steps=args.max_steps if args.max_steps is not None else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=FIRDataCollator(tokenizer=tokenizer),
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] Saved adapter/model + tokenizer to {final_dir}")


if __name__ == "__main__":
    main()
