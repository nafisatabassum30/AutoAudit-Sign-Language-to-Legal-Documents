"""LLM module for converting recognized Bangla Sign Language text into
FIR-ready (First Information Report) legal complaints in Bangla.

This package is the "LLM stage" of the AutoAudit pipeline:

    BdSL video --(ST-GNN)--> raw Bangla text --(this module)--> FIR document

See ``llm/README.md`` for the full pipeline documentation.
"""
