"""
Inference pipeline: raw Bangla text → structured FIR legal complaint.

Supports:
  - Local fine-tuned adapter (LoRA/QLoRA merged or unmerged)
  - Batched generation
  - Streaming output
  - Entity extraction post-processing (victim, suspect, location, time, offense)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    model_path: str = "models/checkpoints/final_adapter"
    base_model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    device: str = "auto"
    merge_weights: bool = False


# ---------------------------------------------------------------------------
# FIR entity extractor
# ---------------------------------------------------------------------------

@dataclass
class FIREntities:
    complainant: str = ""
    accused: str = ""
    location: str = ""
    date: str = ""
    time: str = ""
    offense_type: str = ""
    legal_section: str = ""
    description: str = ""


_FIELD_PATTERNS = {
    "complainant": [
        r"অভিযোগকারীর তথ্য[:\s]*\n\s*নাম[:\s]*([^\n]+)",
        r"নিম্নস্বাক্ষরকারী\s+([^\s]+(?:\s+[^\s]+)?)\s+এতদ্বারা",
    ],
    "accused": [
        r"অভিযুক্তের তথ্য[:\s]*\n\s*নাম[:\s]*([^\n]+)",
        r"অভিযুক্ত\s+([^\s]+(?:\s+[^\s]+)?)\s+",
    ],
    "location": [
        r"ঘটনাস্থল[:\s]*([^\n]+)",
        r"([^\s]+(?:\s+[^\s]+)?)\s+এলাকায়",
    ],
    "date": [
        r"ঘটনার তারিখ ও সময়[:\s]*([0-9/]+)",
        r"তারিখ[:\s]*([0-9/]+)",
    ],
    "time": [
        r"ঘটনার তারিখ ও সময়[:\s]*[0-9/]+,\s*([^\n]+)",
        r"সময়[:\s]*([^\n,]+)",
    ],
    "legal_section": [
        r"প্রযোজ্য আইনি ধারা[:\s]*([^\n]+)",
    ],
    "offense_type": [
        r"অপরাধের ধরন[:\s]*([^\n]+)",
    ],
}


def extract_entities(fir_text: str) -> FIREntities:
    entities = FIREntities()
    for field_name, patterns in _FIELD_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, fir_text, re.UNICODE)
            if match:
                setattr(entities, field_name, match.group(1).strip())
                break
    return entities


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class FIRGenerator:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        try:
            from unsloth import FastLanguageModel  # type: ignore

            logger.info("Loading model via Unsloth from %s", self.config.model_path)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
            )
            FastLanguageModel.for_inference(self.model)

        except ImportError:
            logger.info("Unsloth not found — using HuggingFace PEFT loader")
            self._load_hf()

        self._loaded = True
        logger.info("Model loaded successfully.")

    def _load_hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = None
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        adapter_path = Path(self.config.model_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info("Loading LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            if self.config.merge_weights:
                logger.info("Merging LoRA weights into base model...")
                self.model = self.model.merge_and_unload()
        else:
            logger.warning("No adapter found at %s — using base model only", adapter_path)
            self.model = base_model

        self.model.eval()

    def _build_prompt(self, informal_text: str) -> str:
        from src.data_prep import format_prompt, DEFAULT_INSTRUCTION
        return format_prompt(DEFAULT_INSTRUCTION, informal_text)

    def generate(self, informal_text: str) -> str:
        self.load()
        import torch

        prompt = self._build_prompt(informal_text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        fir_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return fir_text.strip()

    def generate_batch(self, texts: list[str]) -> list[str]:
        return [self.generate(t) for t in texts]

    def generate_structured(self, informal_text: str) -> dict:
        fir_text = self.generate(informal_text)
        entities = extract_entities(fir_text)
        return {
            "input": informal_text,
            "fir_text": fir_text,
            "entities": {
                "complainant": entities.complainant,
                "accused": entities.accused,
                "location": entities.location,
                "date": entities.date,
                "time": entities.time,
                "offense_type": entities.offense_type,
                "legal_section": entities.legal_section,
            },
        }

    def stream(self, informal_text: str) -> Iterator[str]:
        """Yield tokens one-by-one for streaming responses."""
        self.load()
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        prompt = self._build_prompt(informal_text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="BdSL FIR Inference")
    parser.add_argument("--model_path", default="models/checkpoints/final_adapter")
    parser.add_argument("--base_model", default="unsloth/llama-3-8b-bnb-4bit")
    parser.add_argument(
        "--text",
        default="গতকাল রাত ১০টায় মিরপুর ১০ এলাকায় বাদল মিয়া আমার মোবাইল ছিনিয়ে নিয়েছে।",
    )
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    cfg = InferenceConfig(model_path=args.model_path, base_model_name=args.base_model)
    gen = FIRGenerator(cfg)

    if args.stream:
        print("Streaming FIR output:\n")
        for token in gen.stream(args.text):
            print(token, end="", flush=True)
        print()
    else:
        result = gen.generate_structured(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
