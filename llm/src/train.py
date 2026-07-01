"""
QLoRA fine-tuning pipeline for the BdSL → Bangla FIR LLM.

Architecture:
  - Base model : Llama-3-8B (or any causal LM; 4-bit NF4 quantisation via bitsandbytes)
  - PEFT       : QLoRA (LoRA rank 16, alpha 32, dropout 0.05)
  - Trainer    : HuggingFace SFTTrainer (trl)
  - Optimiser  : paged AdamW 8-bit

Quick start (single GPU):
    python src/train.py --config configs/training_config.yaml

Full config options: see TrainingConfig dataclass below.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    trust_remote_code: bool = True
    use_cache: bool = False


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Modules to target — adjust per model architecture
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "data/synthetic"
    max_seq_length: int = 1024

    # Output
    output_dir: str = "models/checkpoints"
    logging_dir: str = "models/logs"

    # Training loop
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3

    # Precision
    fp16: bool = False
    bf16: bool = True

    # Checkpointing & evaluation
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    save_total_limit: int = 2

    # Logging
    logging_steps: int = 50
    report_to: str = "tensorboard"

    # Seed
    seed: int = 42

    # Model & LoRA sub-configs (kept separate for clarity)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)


def load_config(path: str) -> TrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = TrainingConfig()
    for key, val in raw.items():
        if key == "model":
            cfg.model = ModelConfig(**val)
        elif key == "lora":
            cfg.lora = LoRAConfig(**val)
        else:
            setattr(cfg, key, val)
    return cfg


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(cfg: TrainingConfig):
    try:
        from unsloth import FastLanguageModel  # type: ignore

        logger.info("Using Unsloth fast path for %s", cfg.model.model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=cfg.model.load_in_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            target_modules=cfg.lora.target_modules,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
        return model, tokenizer

    except ImportError:
        logger.info("Unsloth not available — falling back to standard HuggingFace + bitsandbytes")
        return _build_hf_model(cfg)


def _build_hf_model(cfg: TrainingConfig):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    compute_dtype = getattr(torch, cfg.model.bnb_4bit_compute_dtype)

    if cfg.model.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
        use_cache=cfg.model.use_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def _formatting_prompts_func(examples: dict) -> list[str]:
    """Used by SFTTrainer's dataset_text_field alternative."""
    from src.data_prep import format_prompt, DEFAULT_INSTRUCTION

    texts = []
    for instruction, inp, out in zip(
        examples.get("instruction", [DEFAULT_INSTRUCTION] * len(examples["input"])),
        examples["input"],
        examples["output"],
    ):
        texts.append(format_prompt(instruction, inp, out))
    return texts


# ---------------------------------------------------------------------------
# Trainer builder
# ---------------------------------------------------------------------------

def build_trainer(cfg: TrainingConfig, model, tokenizer, train_dataset, eval_dataset):
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_dir=cfg.logging_dir,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.evaluation_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=cfg.max_seq_length,
        formatting_func=_formatting_prompts_func,
        args=training_args,
    )
    return trainer


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: TrainingConfig):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from datasets import load_from_disk
    from src.data_prep import load_dataset_dict

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.logging_dir, exist_ok=True)

    logger.info("Training config:\n%s", json.dumps(asdict(cfg), indent=2, default=str))

    # Load raw dataset (JSON format — formatting_func handles prompt construction)
    dataset_dict = load_dataset_dict(cfg.data_dir)

    model, tokenizer = build_model_and_tokenizer(cfg)

    trainer = build_trainer(
        cfg,
        model,
        tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
    )

    logger.info("Starting training...")
    trainer_output = trainer.train()
    logger.info("Training complete. Stats: %s", trainer_output)

    # Save final adapter weights
    final_path = Path(cfg.output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Saved adapter + tokenizer → %s", final_path)

    return trainer_output


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune BdSL FIR LLM with QLoRA")
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to YAML config",
    )
    # Allow inline overrides like --learning_rate 1e-4
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else TrainingConfig()

    # Apply CLI overrides
    for key in ("learning_rate", "num_train_epochs", "data_dir", "output_dir"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    if args.model_name:
        cfg.model.model_name = args.model_name

    train(cfg)
