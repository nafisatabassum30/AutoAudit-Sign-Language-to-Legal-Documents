# AutoAudit-Sign-Language-to-Legal-Documents
An AI-powered system that converts Bangladeshi Sign Language (BdSL) into legal complaints. The pipeline features a Spatial-Temporal Graph Neural Network optimized via Google MediaPipe to track and recognize continuous BdSL gestures, paired with a specialized Large Language Model (LLM) fine-tuned via LoRA/QLoRA that converts the recognized Bangla text into a structured, FIR-ready legal complaint.

## Modules

- [`llm/`](llm/README.md) -- the LLM stage: fine-tunes a causal LLM to turn raw Bangla text (from the sign-recognition stage) into structured FIR (First Information Report) JSON and a submission-ready formal Bangla legal document. Includes data preparation, LoRA/QLoRA training, inference, evaluation, and a Flask API. **Start here** -- see `llm/README.md` for setup and usage.
- Sign-recognition stage (MediaPipe pose estimation + Spatial-Temporal GNN) -- not yet implemented in this repository.

