# AutoAudit-Sign-Language-to-Legal-Documents
An AI-powered The system features a unique two-stage pipeline: a Spatial-Temporal Graph Neural Network optimized via Google MediaPipe to track and recognize continuous BdSL gestures, paired with a specialized Large Language Model (LLM) fine-tuned via QLoRA.

## Repository layout

- [`llm/`](llm/README.md) — the LLM stage: fine-tunes a (Q)LoRA-adapted LLM to turn
  informal Bangla text (from the sign-recognition stage) into a structured,
  validated JSON object and a formatted, FIR-ready legal complaint document.
  Includes a synthetic bootstrap dataset generator, training/inference/evaluation
  scripts, and tests. Start here — see [`llm/README.md`](llm/README.md) for details.
- The ST-GNN sign-recognition stage and the Flask/Flutter integration layers
  are future work, to be added as their own top-level directories.
