# Real FIR templates go here

This directory is where you should drop the **1,000+ real FIR / GD templates**
referenced in the project workflow (collected from BLAST, local police
stations, or public case archives).

Recommended format: one `.json` or `.txt` file per template, or a single
`.jsonl` file with one record per line. Once you have real data here, write a
small converter that maps each template into the same
`{"raw_text": ..., "target": {...FIRComplaint fields...}}` shape used by
`llm/data/generate_dataset.py`, then concatenate it with (or replace) the
synthetic data before running `prepare_dataset.py`. None of the training,
inference, or evaluation code needs to change -- they only depend on the
final JSONL schema, not on where the data came from.

Until real templates are collected, the synthetic generator
(`generate_dataset.py`) plus the hand-authored `seed_examples.jsonl` are used
to bootstrap training so the LLM pipeline can be built, tested, and iterated
on end-to-end.
