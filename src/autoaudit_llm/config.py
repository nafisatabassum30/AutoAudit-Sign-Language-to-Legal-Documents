"""Typed configuration loading for the AutoAudit Bangla FIR LLM.

Configuration is stored as YAML (see ``configs/default.yaml``) and parsed into
lightweight dataclasses so the rest of the codebase gets attribute access and
sane defaults without pulling in a heavy settings framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

T = TypeVar("T")


@dataclass
class ModelConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_dir: str = "outputs/adapter"
    max_seq_length: int = 2048
    load_in_4bit: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/run"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 100
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    seed: int = 42


@dataclass
class DataConfig:
    train_file: str = "data/processed/train.jsonl"
    eval_file: str = "data/processed/eval.jsonl"
    num_synthetic_examples: int = 400
    eval_split: float = 0.1


@dataclass
class InferenceConfig:
    allow_rule_based_fallback: bool = True
    max_new_tokens: int = 768
    temperature: float = 0.3
    top_p: float = 0.9


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass from a plain dict, ignoring extras."""
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _from_dict(f.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load an :class:`AppConfig` from YAML, falling back to defaults.

    Parameters
    ----------
    path:
        Optional path to a YAML config file. When ``None`` the bundled
        ``configs/default.yaml`` is used if present, otherwise built-in
        defaults are returned.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return AppConfig(
        model=_from_dict(ModelConfig, raw.get("model", {})),
        lora=_from_dict(LoraConfig, raw.get("lora", {})),
        training=_from_dict(TrainingConfig, raw.get("training", {})),
        data=_from_dict(DataConfig, raw.get("data", {})),
        inference=_from_dict(InferenceConfig, raw.get("inference", {})),
        api=_from_dict(ApiConfig, raw.get("api", {})),
    )
