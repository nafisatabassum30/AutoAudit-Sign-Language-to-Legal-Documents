"""Inference engine: raw Bangla statement -> :class:`FIRComplaint`.

The :class:`FIRGenerator` prefers the fine-tuned LLM (base model + LoRA
adapter) but transparently falls back to the deterministic rule-based builder
when model weights, GPU or ML libraries are unavailable. This keeps the whole
AutoAudit pipeline runnable in any environment while still supporting the full
fine-tuned model in production.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import AppConfig, load_config
from .prompts import build_chat_messages
from .postprocess import parse_model_output
from .rule_based import build_complaint as rule_based_complaint
from .schema import FIRComplaint

logger = logging.getLogger(__name__)


class FIRGenerator:
    """Generate FIR complaints from raw Bangla statements.

    Parameters
    ----------
    config:
        Application config. Defaults to :func:`load_config`.
    prefer_model:
        When ``True`` (default) the engine attempts to load the fine-tuned
        model. Set ``False`` to force the rule-based baseline (useful for
        tests, CI and low-resource deployments).
    """

    def __init__(self, config: Optional[AppConfig] = None, prefer_model: bool = True):
        self.config = config or load_config()
        self.prefer_model = prefer_model
        self._model = None
        self._tokenizer = None
        self._model_ready = False
        if prefer_model:
            self._try_load_model()

    # -- model loading -------------------------------------------------------
    def _try_load_model(self) -> None:
        """Attempt to load base model + LoRA adapter; degrade gracefully."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("ML libraries unavailable (%s); using rule-based fallback.", exc)
            return

        adapter_dir = Path(self.config.model.adapter_dir)
        base_model = self.config.model.base_model

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto"
            )
            if adapter_dir.exists():
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter_dir))
                logger.info("Loaded LoRA adapter from %s", adapter_dir)
            else:
                logger.warning(
                    "Adapter dir %s not found; using base model without fine-tuning.",
                    adapter_dir,
                )
            model.eval()
            self._model = model
            self._model_ready = True
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("Failed to load model (%s); using rule-based fallback.", exc)
            self._model = None
            self._model_ready = False

    @property
    def using_model(self) -> bool:
        return self._model_ready

    # -- generation ----------------------------------------------------------
    def _generate_with_model(self, raw_statement: str) -> Optional[str]:
        import torch

        messages = build_chat_messages(raw_statement)
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover - most instruct models have a template
            prompt = "\n".join(m["content"] for m in messages)

        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.inference.max_new_tokens,
                temperature=self.config.inference.temperature,
                top_p=self.config.inference.top_p,
                do_sample=self.config.inference.temperature > 0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return tok.decode(generated, skip_special_tokens=True)

    def generate(self, raw_statement: str) -> FIRComplaint:
        """Convert a raw Bangla statement into a validated FIR complaint."""
        if not raw_statement or not raw_statement.strip():
            raise ValueError("raw_statement must be a non-empty string")

        if self._model_ready:
            try:
                text = self._generate_with_model(raw_statement)
                complaint = parse_model_output(text or "")
                if complaint is not None:
                    return complaint
                logger.warning("Model output unparseable; falling back to rules.")
            except Exception as exc:  # pragma: no cover - env dependent
                logger.warning("Model generation failed (%s); falling back.", exc)

        if not self.config.inference.allow_rule_based_fallback and self.prefer_model:
            raise RuntimeError(
                "Model unavailable and rule-based fallback disabled by config."
            )
        return rule_based_complaint(raw_statement)
