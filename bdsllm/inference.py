from __future__ import annotations

from dataclasses import dataclass

from bdsllm.prompts import build_chat_messages, build_instruction
from bdsllm.schema import IncidentFacts
from bdsllm.templates import render_legal_complaint


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 900
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class BanglaLegalComplaintGenerator:
    """Generate FIR-style Bangla complaints with a fine-tuned model or fallback template."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        adapter_path: str | None = None,
        device_map: str = "auto",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        if not self.model_name_or_path:
            return

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self.device_map,
            torch_dtype="auto",
        )
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def generate(self, facts: IncidentFacts, config: GenerationConfig | None = None) -> str:
        if not self.is_model_loaded:
            self.load()
        if not self.is_model_loaded:
            return render_legal_complaint(facts)

        config = config or GenerationConfig()
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None
        assert model is not None

        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                build_chat_messages(facts),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{build_instruction(facts)}\n\nউত্তর:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
