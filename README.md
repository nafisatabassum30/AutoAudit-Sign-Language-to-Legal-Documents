# AutoAudit-Sign-Language-to-Legal-Documents
An AI-powered The system features a unique two-stage pipeline: a Spatial-Temporal Graph Neural Network optimized via Google MediaPipe to track and recognize continuous BdSL gestures, paired with a specialized Large Language Model (LLM) fine-tuned via QLoRA.

## Pipeline

```
BdSL video --(MediaPipe + ST-GNN)--> raw Bangla text --(LLM)--> FIR-ready legal complaint
```

## LLM stage

The Bangla-text-to-FIR LLM stage (fine-tuning, inference, Flask API, and
evaluation) lives in [`llm/`](llm/README.md). It includes a synthetic
bootstrap dataset generator so the full pipeline can be built and tested
end-to-end before real FIR templates are collected.

The Spatial-Temporal GNN sign-recognition stage is not yet implemented in
this repository.

