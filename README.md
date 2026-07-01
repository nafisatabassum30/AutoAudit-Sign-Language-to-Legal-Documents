# AutoAudit-Sign-Language-to-Legal-Documents

An AI-powered system for converting Bangladeshi Sign Language (BdSL) into
FIR-ready legal complaints. The system features a two-stage pipeline: a
Spatial-Temporal Graph Neural Network optimized via Google MediaPipe to track
and recognize continuous BdSL gestures, paired with a specialized Large
Language Model (LLM) fine-tuned via QLoRA that converts the recognized raw
Bangla text into a structured, formal FIR legal complaint document.

## Pipeline

1. **Input**: User signs into a camera (no specialized hardware required).
2. **MediaPipe Pose Estimation**: extracts skeletal keypoints per frame.
3. **Spatial-Temporal GNN**: recognizes BdSL words/sentences from keypoint
   sequences and outputs raw Bangla text.
4. **LLM Conversion** (this repo's [`llm/`](./llm) module): a QLoRA fine-tuned
   LLM turns the raw Bangla text into a structured, FIR-ready legal complaint.
5. **FIR-Ready Legal Complaint**: a complete document with date, time,
   location, offense description, and victim/complainant information, ready
   for review and submission to law enforcement.

## Repository layout

- [`llm/`](./llm) — the LLM stage: prompt design, synthetic + real dataset
  tooling, QLoRA fine-tuning, inference, evaluation, and a Flask API bridging
  this stage to the rest of the pipeline. See [`llm/README.md`](./llm/README.md)
  for setup and usage instructions.
- The MediaPipe / Spatial-Temporal GNN sign-recognition stage and the mobile
  app are tracked as future work in this repository.

