"""Fine-tune a causal LLM (LoRA or QLoRA) to convert raw Bangla text
into structured FIR JSON.

This script is the "Fine-tune Llama 3 (LoRA) on legal templates" stage
of the AutoAudit pipeline. It is deliberately backend-agnostic: point
it at any Hugging Face causal LM (Qwen2.5, Llama-3.x, Gemma-2, ...) via
``--model-name-or-path`` and it will:

  1. Load & prepare the Bangla-FIR dataset (data_preparation.py).
  2. Load the base model, optionally in 4-bit (QLoRA) on a GPU.
  3. Attach a LoRA adapter (peft).
  4. Fine-tune with the HF ``Trainer`` API, masking the prompt tokens
     out of the loss so only FIR-JSON generation is trained.
  5. Save the adapter (and tokenizer) to ``--output-dir``.

Example (GPU, QLoRA):
    python -m src.train --config configs/training_config.yaml

Example (CPU smoke test, tiny model, no quantization):
    python -m src.train \\
        --model-name-or-path sshleifer/tiny-gpt2 \\
        --no-4bit --epochs 1 --max-steps 5 \\
        --output-dir outputs/smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from .data_preparation import prepare_dataset
except ImportError:  # pragma: no cover
    from data_preparation import prepare_dataset

LLM_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        path = LLM_ROOT / "configs" / "training_config.yaml"
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to a YAML training config.")
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--raw-data", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N steps (overrides epochs); useful for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--4bit", dest="use_4bit", action="store_true", default=None, help="Force-enable 4-bit QLoRA (requires CUDA + bitsandbytes).")
    parser.add_argument("--no-4bit", dest="use_4bit", action="store_false", help="Force-disable 4-bit quantization (use plain LoRA).")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)

    model_name = args.model_name_or_path or cfg["model"]["name_or_path"]
    use_4bit = cfg["model"]["load_in_4bit"] if args.use_4bit is None else args.use_4bit
    use_4bit = use_4bit and torch.cuda.is_available()

    raw_path = args.raw_data or (LLM_ROOT / cfg["data"]["raw_path"])
    processed_dir = args.processed_dir or (LLM_ROOT / cfg["data"]["processed_dir"])
    output_dir = args.output_dir or (LLM_ROOT / cfg["training"]["output_dir"])
    max_length = args.max_length or cfg["data"]["max_length"]

    return {
        "model_name": model_name,
        "use_4bit": use_4bit,
        "raw_path": Path(raw_path),
        "processed_dir": Path(processed_dir),
        "output_dir": Path(output_dir),
        "max_length": max_length,
        "val_ratio": cfg["data"]["val_ratio"],
        "seed": args.seed or cfg["data"]["seed"],
        "epochs": args.epochs if args.epochs is not None else cfg["training"]["num_train_epochs"],
        "batch_size": args.batch_size or cfg["training"]["per_device_train_batch_size"],
        "eval_batch_size": cfg["training"]["per_device_eval_batch_size"],
        "grad_accum": args.grad_accum or cfg["training"]["gradient_accumulation_steps"],
        "lr": args.lr or cfg["training"]["learning_rate"],
        "lora": cfg["lora"],
        "bf16": cfg["training"]["bf16"] and torch.cuda.is_available(),
        "gradient_checkpointing": cfg["training"]["gradient_checkpointing"],
        "max_steps": args.max_steps,
        "cfg": cfg,
    }


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_model(settings: Dict[str, Any], trust_remote_code: bool = False):
    quantization_config = None
    torch_dtype = torch.bfloat16 if settings["bf16"] else torch.float32

    if settings["use_4bit"]:
        from transformers import BitsAndBytesConfig

        model_cfg = settings["cfg"]["model"]
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=model_cfg["use_double_quant"],
        )

    model = AutoModelForCausalLM.from_pretrained(
        settings["model_name"],
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return model


def attach_lora(model, lora_cfg: Dict[str, Any], use_4bit: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = lora_cfg.get("target_modules", "all-linear")
    try:
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
    except ValueError:
        # Fall back to a conservative, explicit module list for older
        # peft versions or architectures without "all-linear" support.
        fallback_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=fallback_modules,
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model


class JsonlCausalLMDataset(torch.utils.data.Dataset):
    """Loads pre-tokenized {input_ids, attention_mask, labels} records."""

    def __init__(self, path: Path):
        self.records: List[Dict[str, List[int]]] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        return {
            "input_ids": torch.tensor(rec["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(rec["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(rec["labels"], dtype=torch.long),
        }


def make_collate_fn(pad_token_id: int):
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)

        def pad(seq: torch.Tensor, value: int) -> torch.Tensor:
            pad_len = max_len - seq.size(0)
            if pad_len <= 0:
                return seq
            return torch.cat([seq, torch.full((pad_len,), value, dtype=seq.dtype)])

        return {
            "input_ids": torch.stack([pad(item["input_ids"], pad_token_id) for item in batch]),
            "attention_mask": torch.stack([pad(item["attention_mask"], 0) for item in batch]),
            "labels": torch.stack([pad(item["labels"], -100) for item in batch]),
        }

    return collate


def main() -> None:
    args = build_arg_parser().parse_args()
    settings = resolve_settings(args)

    trust_remote_code = settings["cfg"]["model"].get("trust_remote_code", False)
    tokenizer = load_tokenizer(settings["model_name"], trust_remote_code=trust_remote_code)

    print(f"Preparing dataset from {settings['raw_path']} -> {settings['processed_dir']}")
    counts = prepare_dataset(
        settings["raw_path"],
        tokenizer,
        settings["processed_dir"],
        val_ratio=settings["val_ratio"],
        max_length=settings["max_length"],
        seed=settings["seed"],
    )
    print(f"Dataset ready: {counts}")

    train_dataset = JsonlCausalLMDataset(settings["processed_dir"] / "train_tokenized.jsonl")
    eval_dataset = JsonlCausalLMDataset(settings["processed_dir"] / "val_tokenized.jsonl")

    print(f"Loading base model: {settings['model_name']} (4bit={settings['use_4bit']})")
    model = load_model(settings, trust_remote_code=trust_remote_code)
    model = attach_lora(model, settings["lora"], settings["use_4bit"])

    output_dir = settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings["epochs"],
        max_steps=settings["max_steps"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["eval_batch_size"],
        gradient_accumulation_steps=settings["grad_accum"],
        learning_rate=settings["lr"],
        lr_scheduler_type=settings["cfg"]["training"]["lr_scheduler_type"],
        warmup_ratio=settings["cfg"]["training"]["warmup_ratio"],
        weight_decay=settings["cfg"]["training"]["weight_decay"],
        logging_steps=settings["cfg"]["training"]["logging_steps"],
        eval_strategy=settings["cfg"]["training"]["eval_strategy"],
        save_strategy=settings["cfg"]["training"]["save_strategy"],
        save_total_limit=settings["cfg"]["training"]["save_total_limit"],
        bf16=settings["bf16"],
        gradient_checkpointing=settings["gradient_checkpointing"] and torch.cuda.is_available(),
        report_to=settings["cfg"]["training"]["report_to"],
        seed=settings["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
    )

    trainer.train()

    print(f"Saving LoRA adapter + tokenizer to {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
