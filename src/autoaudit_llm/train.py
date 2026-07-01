"""QLoRA fine-tuning of the Bangla FIR LLM.

Fine-tunes a base causal LM (e.g. Llama-3-8B-Instruct) with 4-bit quantization
(QLoRA) on the instruction dataset produced by :mod:`autoaudit_llm.dataset`.

This module imports torch/transformers/peft/trl lazily so importing the package
never requires the heavy training stack. Run it via::

    python -m autoaudit_llm.cli train --config configs/default.yaml

A CUDA-capable GPU is strongly recommended.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import AppConfig, load_config
from .dataset import build_hf_dataset, read_jsonl

logger = logging.getLogger(__name__)


def train(config: AppConfig | None = None) -> str:
    """Run QLoRA fine-tuning and return the saved adapter directory."""
    config = config or load_config()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

    train_records = read_jsonl(config.data.train_file)
    eval_records = (
        read_jsonl(config.data.eval_file)
        if Path(config.data.eval_file).exists()
        else []
    )
    logger.info("Loaded %d train / %d eval examples", len(train_records), len(eval_records))

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if config.model.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = build_hf_dataset(train_records, tokenizer)
    eval_ds = build_hf_dataset(eval_records, tokenizer) if eval_records else None

    trainer = _build_trainer(config, model, tokenizer, train_ds, eval_ds)
    trainer.train()

    adapter_dir = config.model.adapter_dir
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved LoRA adapter to %s", adapter_dir)
    return adapter_dir


def _build_trainer(config, model, tokenizer, train_ds, eval_ds):
    """Construct a TRL SFTTrainer, tolerating API differences across versions."""
    from trl import SFTTrainer, SFTConfig

    sft_config = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        bf16=config.training.bf16,
        seed=config.training.seed,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        report_to=[],
    )
    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
