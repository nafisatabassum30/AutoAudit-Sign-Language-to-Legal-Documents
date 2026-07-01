"""AutoAudit Bangla FIR LLM package.

This package implements the *language* stage of the AutoAudit pipeline that
converts Bangladeshi Sign Language (BdSL) into FIR-ready legal complaints:

    BdSL video --(MediaPipe + ST-GNN)--> raw Bangla text
                                             |
                                             v
                        [ autoaudit_llm ]  fine-tuned LLM
                                             |
                                             v
                              structured, FIR-ready complaint (Bangla)

The heavy ML dependencies (torch/transformers/peft) are only imported lazily
inside the training/inference modules, so the schema, prompt, dataset and
rule-based components can be used without a GPU.
"""

from .schema import FIRComplaint, FIRExtraction
from .config import AppConfig, load_config

__all__ = [
    "FIRComplaint",
    "FIRExtraction",
    "AppConfig",
    "load_config",
    "__version__",
]

__version__ = "0.1.0"
